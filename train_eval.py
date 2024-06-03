import logging
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from src.bart import MyConfig
from src.utils import DatasetConfig


def move_to_device(batch_data, device='cpu', generate=False):
    # with labels
    not_include_in_generation = ['labels']
    model_inputs = {}
    clip_value = {}
    image_number = {}
    summary = None
    for key in batch_data:
        if generate and (key in not_include_in_generation):
            continue
        elif key == 'summary':
            summary = batch_data['summary']
        elif key == 'sentence_clip' or key == 'summary_clip':
            clip_value[key] = [i.to(device) for i in batch_data[key]]
        elif key == 'image_number' or key == 'target_image_number':
            image_number[key] = [i for i in batch_data[key]]
        else:
            model_inputs[key] = batch_data[key].to(device)
    return model_inputs, summary, clip_value, image_number


def cal_kl_div(pred, target, temperature = 10, data_dim = 0):
    pred = F.softmax(pred / temperature, dim=data_dim).log()
    target = F.softmax(target / temperature, dim=data_dim)
    kl_div = F.kl_div(pred, target, reduction='sum')
    return kl_div


def print_config(my_config: MyConfig, dataset_config: DatasetConfig):
    logger_config = {}
    logger_config['my_config'] = my_config.__dict__
    logger_config['dataset_config'] = dataset_config.__dict__
    return json.dumps(logger_config, indent=4)


def train(logger, my_config, dataset_config, model, train_iter, valid_iter=None, test_iter=None, cur_epoch='0'):
    logger.info(print_config(my_config, dataset_config))
    model = model.to(my_config.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=my_config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=my_config.start_factor,
        end_factor=my_config.end_factor,
        total_iters=my_config.warm_up_step_num,
    )
    torch.optim.AdamW(model.parameters(), lr=my_config.learning_rate, )

    logger.info('Training')
    step = 0
    total_step = len(train_iter)


    epoch = int(cur_epoch)
    logger.info(f'Epoch {epoch}:')
    for batch_data in tqdm(train_iter):
        model_inputs, summary, clip_value, _ = move_to_device(batch_data, my_config.device)
        outputs = model(**model_inputs, return_dict=False)
        lm_loss = outputs[0]
        total_loss = lm_loss * my_config.lm_loss_weight

        # if my_config.use_kl_div:
        #     # sentence clip
        #     sentence_clip = clip_value['sentence_clip']
        #     image_sentence_values = outputs[3]
        #     temperature = 10
        #     for i, image_sentence_value in enumerate(image_sentence_values):
        #         kl_div = cal_kl_div(image_sentence_value, sentence_clip[i], data_dim=1)
        #         total_loss += kl_div
        # image select
        if my_config.has_image_select:
            summary_clip = clip_value['summary_clip']
            image_select_values = outputs[-1]
            for i, image_select_value in enumerate(image_select_values):
                kl_div = cal_kl_div(image_select_value, summary_clip[i], data_dim=0)
                total_loss += kl_div * my_config.is_loss_weight

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        step += 1
        if step % my_config.log_step == 0:
            logger.info('Batch {0:>6}/{1}: train_loss: {2:>.2}'.format
                        (step, total_step, total_loss.cpu().detach().numpy()))

        # if step % my_config.save_step == 0 or step % total_step == 0:
        #     torch.save(model.state_dict(), f'{my_config.save_dir}/model_step_{step:0>6}.bin')

        if step % total_step == 0:
            torch.save(model.state_dict(), f'{my_config.save_dir}/{dataset_config.dataset}_epoch{epoch}.bin')
        # if step % my_config.save_step == 0:
        #     torch.save(model.state_dict(), f'{my_config.save_dir}/model_step_{step:0>6}.bin')


def test(logger, my_config, model, test_iter, hyp_file=None, ref_file=None):
    from transformers import BartTokenizer
    from rouge import Rouge
    tokenizer = BartTokenizer.from_pretrained(my_config.bart_model)

    model = model.to(my_config.device)

    logger.info('Testing')
    model.eval()
    # summarization
    hyps = []
    refs = []
    # image select
    has_image_num = 0
    has_target_image_num = 0
    get_target_image_num = 0
    top_3_get_target_image_num = 0
    for batch_data in tqdm(test_iter):
        generation_inputs, summary, clip_value, image_number = move_to_device(batch_data, my_config.device, generate=True)
        # generation_outputs = model.generate(**generation_inputs, num_beams=5)
        generation_outputs = model.generate(**generation_inputs, num_beams=5, min_length=my_config.min_length, max_length=my_config.max_length, length_penalty=my_config.length_penalty)
        # summarization
        pred_summary = tokenizer.batch_decode(generation_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        hyps.extend(pred_summary)
        refs.extend(summary)
        # image select
        image_scores = generation_outputs[1][0]
        for i, image_score in enumerate(image_scores):
            src_number_list = image_number['image_number'][i]
            tgt_number_list = image_number['target_image_number'][i]
            if tgt_number_list is None or len(tgt_number_list) == 0:
                continue
            has_image_num += 1
            has_target = bool(set(src_number_list) & set(tgt_number_list))
            if has_target:
                has_target_image_num += 1
            select_image_number = src_number_list[image_score.argmax()]
            if select_image_number in tgt_number_list:
                get_target_image_num += 1
            if select_image_number in tgt_number_list[:3]:
                top_3_get_target_image_num += 1


    if hyp_file is not None:
        with open(hyp_file, 'w') as fw:
            for hyp in hyps:
                fw.write(hyp)
                fw.write('\n')
    if ref_file is not None:
        with open(ref_file, 'w') as fw:
            for ref in refs:
                fw.write(ref)
                fw.write('\n')
    rouge = Rouge()
    scores = rouge.get_scores(hyps, refs, avg=True)
    logger.info('Test ROUGE')
    logger.info(json.dumps(scores, indent=4))

    logger.info('Image Precision')
    logger.info(f'{has_image_num} data items have target images.')
    logger.info(f'{has_target_image_num} data items have selected at least one target images as input.')
    logger.info(f'{get_target_image_num} data items have correctly picked out at least one target image.')
    logger.info(f'{get_target_image_num} / {has_image_num} = {get_target_image_num / has_image_num}')
    logger.info(f'{get_target_image_num} / {has_target_image_num} = {get_target_image_num / has_target_image_num}')

    logger.info(f'{top_3_get_target_image_num} data items have correctly picked out at least one of the first three target images.')
    logger.info(f'{top_3_get_target_image_num} / {has_image_num} = {top_3_get_target_image_num / has_image_num}')
    logger.info(f'{top_3_get_target_image_num} / {has_target_image_num} = {top_3_get_target_image_num / has_target_image_num}')
    
    return scores


def get_loss(logger, my_config, dataset_config, model, data_iter):
    logger.info('LOSS ONLY!!')
    logger.info(print_config(my_config, dataset_config))
    model = model.to(my_config.device)
    model.eval()
    logger.info('Training')
    dataset_loss = 0
    step = 0
    total_step = len(data_iter)

    for batch_data in tqdm(data_iter):
        model_inputs, summary, clip_value, _ = move_to_device(batch_data, my_config.device)
        outputs = model(**model_inputs, return_dict=False)
        lm_loss = outputs[0]
        total_loss = lm_loss * my_config.lm_loss_weight

        # if my_config.use_kl_div:
        #     # sentence clip
        #     sentence_clip = clip_value['sentence_clip']
        #     image_sentence_values = outputs[3]
        #     temperature = 10
        #     for i, image_sentence_value in enumerate(image_sentence_values):
        #         kl_div = cal_kl_div(image_sentence_value, sentence_clip[i], data_dim=1)
        #         total_loss += kl_div
        # image select
        if my_config.has_image_select:
            summary_clip = clip_value['summary_clip']
            image_select_values = outputs[-1]
            for i, image_select_value in enumerate(image_select_values):
                kl_div = cal_kl_div(image_select_value, summary_clip[i], data_dim=0)
                total_loss += kl_div * my_config.is_loss_weight

        dataset_loss += total_loss.cpu().detach().numpy()

        step += 1
        if step % my_config.log_step == 0:
            logger.info('Batch {0:>6}/{1}: loss: {2:>.2}'.format
                        (step, total_step, total_loss.cpu().detach().numpy()))
    return dataset_loss
    

def human_eval(logger, my_config, model, test_iter, hyp_file=None, ref_file=None, img_file=None):
    from transformers import BartTokenizer
    from rouge import Rouge
    tokenizer = BartTokenizer.from_pretrained(my_config.bart_model)

    model = model.to(my_config.device)

    logger.info('Testing')
    model.eval()
    # summarization
    hyps = []
    refs = []
    imgs = []
    # image select
    has_image_num = 0
    has_target_image_num = 0
    get_target_image_num = 0
    top_3_get_target_image_num = 0
    for batch_data in tqdm(test_iter):
        generation_inputs, summary, clip_value, image_number = move_to_device(batch_data, my_config.device, generate=True)
        # generation_outputs = model.generate(**generation_inputs, num_beams=5)
        generation_outputs = model.generate(**generation_inputs, num_beams=5, min_length=my_config.min_length, max_length=my_config.max_length, length_penalty=my_config.length_penalty)
        # summarization
        pred_summary = tokenizer.batch_decode(generation_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        hyps.extend(pred_summary)
        refs.extend(summary)
        # image select
        image_scores = generation_outputs[1][0]
        for i, image_score in enumerate(image_scores):
            src_number_list = image_number['image_number'][i]
            tgt_number_list = image_number['target_image_number'][i]
            if tgt_number_list is None or len(tgt_number_list) == 0:
                continue
            has_image_num += 1
            has_target = bool(set(src_number_list) & set(tgt_number_list))
            if has_target:
                has_target_image_num += 1
            select_image_number = src_number_list[image_score.argmax()]
            if select_image_number in tgt_number_list:
                get_target_image_num += 1
            if select_image_number in tgt_number_list[:3]:
                top_3_get_target_image_num += 1
            imgs.append(select_image_number)


    if hyp_file is not None:
        with open(hyp_file, 'w') as fw:
            for hyp in hyps:
                fw.write(hyp)
                fw.write('\n')
    if ref_file is not None:
        with open(ref_file, 'w') as fw:
            for ref in refs:
                fw.write(ref)
                fw.write('\n')
    if img_file is not None:
        with open(img_file, 'w') as fw:
            for img in imgs:
                fw.write(img)
                fw.write('\n')
    rouge = Rouge()
    scores = rouge.get_scores(hyps, refs, avg=True)
    logger.info('Test ROUGE')
    logger.info(json.dumps(scores, indent=4))

    logger.info('Image Precision')
    logger.info(f'{has_image_num} data items have target images.')
    logger.info(f'{has_target_image_num} data items have selected at least one target images as input.')
    logger.info(f'{get_target_image_num} data items have correctly picked out at least one target image.')
    logger.info(f'{get_target_image_num} / {has_image_num} = {get_target_image_num / has_image_num}')
    logger.info(f'{get_target_image_num} / {has_target_image_num} = {get_target_image_num / has_target_image_num}')

    logger.info(f'{top_3_get_target_image_num} data items have correctly picked out at least one of the first three target images.')
    logger.info(f'{top_3_get_target_image_num} / {has_image_num} = {top_3_get_target_image_num / has_image_num}')
    logger.info(f'{top_3_get_target_image_num} / {has_target_image_num} = {top_3_get_target_image_num / has_target_image_num}')
    
    return scores
