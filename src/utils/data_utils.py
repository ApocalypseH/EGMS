import torch
import pickle
import random
import numpy as np

from torch.utils.data import Dataset
from transformers import CLIPProcessor, BartTokenizer
from PIL import Image

from .dataset_configuration import DatasetConfig
from transformers.models.bart.configuration_bart import BartConfig


class MyDataset(Dataset):
    def __init__(self, dataset_config: DatasetConfig, bart_config: BartConfig, my_config, is_test = False, is_valid = False) -> None:
        super().__init__()
        self.is_test = is_test
        self.is_valid = is_valid
        self.src_file = dataset_config.new_src_file

        self.processor = CLIPProcessor.from_pretrained(my_config.clip_model)
        self.tokenizer = BartTokenizer.from_pretrained(my_config.bart_model)
        self.pad_token_id = bart_config.pad_token_id
        self.decoder_start_token_id = bart_config.decoder_start_token_id

        ###############
        self.inline_entity = my_config.inline_entity
        self.max_inline_entity_num = my_config.max_inline_entity_num
        self.max_inline_scene_num = my_config.max_inline_scene_num

        # self.image_size = my_config.image_size
        self.total_len = bart_config.max_position_embeddings
        # self.per_image_len = int((my_config.image_size / my_config.patch_size) ** 2 + 1) # cls emb
        self.blip_query_num = my_config.blip_query_num
        self.per_image_len = my_config.blip_query_num + 1
        self.max_image_num = my_config.max_image_num
        self.image_len = self.per_image_len * my_config.max_image_num
        self.text_len = self.total_len - self.image_len

        if self.inline_entity:
            self.text_len -= self.max_inline_entity_num + self.max_inline_scene_num

        with open(self.src_file, 'rb') as fr:
            self.data_dict = pickle.load(fr)
        self.index2key = list(self.data_dict.keys())
        self.img_dir = dataset_config.img_dir
        self.d_image = my_config.d_image

        self.d_entity = my_config.d_entity
        # self.max_entity_num = my_config.max_entity_num
        # self.max_scene_num = my_config.max_scene_num

        # self.entity_num = 256 + 128
        # self.scene_num = bart_config.max_position_embeddings - self.image_len - self.entity_num
        self.max_entity_len = bart_config.max_position_embeddings - self.image_len

        self.entity_pad_token_id = my_config.entity_pad_token_id
        self.scene_token_id = my_config.scene_token_id
        self.entity_id_offset = my_config.entity_id_offset

        self.ent2idx = {}
        with open(dataset_config.kg_ent2idx_file, 'r', encoding='utf-8') as fr:
            fr.readline()
            for line in fr.readlines():
                ent, idx = line.strip().split('\t')
                self.ent2idx[ent] = int(idx)

    # def sample_entity(self, data_item_dict, *key, max_entity_num=256, max_scene_num=32):
    #     all_ent_list = []
    #     sent_ent_list = []
    #     data = []
    #     for k in key:
    #         data.extend(data_item_dict[k])
    #     for sent in data:
    #         ent_list = []
    #         for ent in sent:
    #             idx = self.ent2idx[ent]
    #             ent_list.append(idx)
    #         all_ent_list.extend(ent_list)
    #         sent_ent_list.append(ent_list)
    #     ent_sample_num = min(max_entity_num, len(all_ent_list))
    #     all_ent_sample = random.sample(all_ent_list, ent_sample_num)
    #     all_ent_attention_mask = torch.zeros(max_entity_num, dtype=torch.int64)
    #     all_ent_attention_mask[:ent_sample_num] = 1
    #     all_ent_embeds = torch.zeros(max_entity_num, self.d_entity)
    #     all_ent_embeds[:ent_sample_num, :] = self.kg_emb[torch.LongTensor(all_ent_sample)]

    #     sent_sample_num = min(max_scene_num, len(sent_ent_list))
    #     all_sent_ent_sample = sent_ent_list[:sent_sample_num]
    #     all_sent_ent_attention_mask = torch.zeros(max_scene_num, dtype=torch.int64)
    #     all_sent_ent_attention_mask[:sent_sample_num] = 1
    #     all_sent_ent_embeds = torch.zeros(max_scene_num, self.d_entity)
    #     for i in range(sent_sample_num):
    #         if len(all_sent_ent_sample[i]) > 0:
    #             sent_ent_embeds = self.kg_emb[torch.LongTensor(all_sent_ent_sample[i])]
    #             all_sent_ent_embeds[i] = sent_ent_embeds.mean(dim=0, keepdims=False)
    #         else: # this sentence has no entity
    #             all_sent_ent_embeds[i] = torch.zeros_like(self.kg_emb[0])
    #             all_ent_attention_mask[i] = 0
    #     return all_ent_embeds, all_ent_attention_mask, all_sent_ent_embeds, all_sent_ent_attention_mask
    
    def get_entity(self, data_item_dict, *key, max_entity_len = 128):
        input_ids = torch.ones(max_entity_len, dtype=torch.int64) * self.entity_pad_token_id
        attention_mask = torch.zeros(max_entity_len, dtype=torch.int64)
        data = []
        count = 0
        for k in key:
            data.extend(data_item_dict[k])
        for sent in data:
            # scence token
            if count >= max_entity_len:
                break
            input_ids[count] = self.scene_token_id
            attention_mask[count] = 1
            count += 1
            # entity token
            for ent in sent:
                if count >= max_entity_len:
                    break
                idx = self.ent2idx.get(ent)
                if idx is not None:
                    input_ids[count] = idx + self.entity_id_offset
                    attention_mask[count] = 1
                    count += 1
        return input_ids, attention_mask

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index: int):
        key = self.index2key[index]
        data_item_dict = self.data_dict[key]
        return_dict = {}

        ########### TEXT-encoder ###########
        ## input_ids
        text_list = data_item_dict['title'] + data_item_dict['body']
        text = '</s><s>'.join(text_list)
        tokenized_dict = self.tokenizer([text], 
                                        return_tensors='pt', 
                                        padding='max_length', 
                                        truncation=True, 
                                        max_length=self.text_len)
        input_ids = tokenized_dict['input_ids'][0]
        return_dict['input_ids'] = input_ids
        ## attention_mask
        attention_mask = tokenized_dict['attention_mask'][0]
        return_dict['attention_mask'] = attention_mask
        ## text_cls_mask
        text_cls_mask = (input_ids == 0).type(torch.int64)
        return_dict['text_cls_mask'] = text_cls_mask
        sentence_num = text_cls_mask.sum().int()

        ## kg_encoder_inputs_embeds
        ## kg_encoder_attention_mask
        ## sent_kg_encoder_inputs_embeds
        ## sent_kg_encoder_attention_mask
        # kg_encoder_inputs_embeds, kg_encoder_attention_mask, sent_kg_encoder_inputs_embeds, sent_kg_encoder_attention_mask = \
        #     self.sample_entity(data_item_dict, 'title_entity', 'body_entity', max_entity_num=self.max_entity_num, max_scene_num=self.max_scene_num)
        # return_dict['kg_encoder_inputs_embeds'] = kg_encoder_inputs_embeds
        # return_dict['kg_encoder_attention_mask'] = kg_encoder_attention_mask
        # return_dict['sent_kg_encoder_inputs_embeds'] = sent_kg_encoder_inputs_embeds
        # return_dict['sent_kg_encoder_attention_mask'] = sent_kg_encoder_attention_mask

        ########### TEXT-decoder ###########
        ## summary
        summary = ' . '.join(data_item_dict['summary']) + ' .'
        return_dict['summary'] = summary
        ## decoder_input_ids
        # tokenized_summary = self.tokenizer([summary], 
        #                                    return_tensors='pt', 
        #                                    padding='max_length', 
        #                                    truncation=True, 
        #                                    max_length=self.total_len)
        # decoder_input_ids = tokenized_summary['input_ids'][0]
        # decoder_input_ids[0] = self.decoder_start_token_id
        # return_dict['decoder_input_ids'] = decoder_input_ids
        # ## decoder_attention_mask
        # decoder_attention_mask = tokenized_summary['attention_mask'][0]
        # return_dict['decoder_attention_mask'] = decoder_attention_mask
        # ## labels
        # labels = decoder_input_ids.new_zeros(decoder_input_ids.shape)
        # labels[:-1] = decoder_input_ids[1:]
        # labels[-1] = self.pad_token_id
        # return_dict['labels'] = labels
        labels = self.tokenizer(
            [summary],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.total_len
        )
        return_dict['labels'] = labels['input_ids'][0]

        ########### IMAGE ###########
        ## image_input
        imgs = data_item_dict['imgs']
        num_imgs = len(imgs)
        if num_imgs > 0:
            # images = [Image.open(f'{self.img_dir}/{key}_{i}.jpg') for i in imgs[:self.max_image_num]]
            # image_input = self.processor(images=images, return_tensors="pt", padding=True)['pixel_values']  # [img_num: 3: 224: 224]
            # image_num = image_input.shape[0]
            # pad_image_input = torch.zeros(self.max_image_num - image_num, 3, self.image_size, self.image_size)
            # image_input = torch.cat([image_input, pad_image_input], dim=0)
            images = data_item_dict['multimodal_img_feature'][:self.max_image_num]
            image_input = torch.cat(images, dim=0)
            image_num = image_input.shape[0]
            pad_image_input = torch.zeros(self.max_image_num - image_num, self.blip_query_num, self.d_image)
            image_input = torch.cat([image_input, pad_image_input], dim=0)
            if self.is_test:
                return_dict['image_number'] = imgs[:image_num]
                return_dict['target_image_number'] = data_item_dict['target_imgs']
            elif self.is_valid:
                return_dict['image_number'] = imgs[:image_num]
        else:
            # image_input = torch.zeros(self.max_image_num, 3, self.image_size, self.image_size)
            image_input = torch.zeros(self.max_image_num, self.blip_query_num, self.d_image)
            image_num = 0
            if self.is_test:
                return_dict['image_number'] = None
                return_dict['target_image_number'] = None
            elif self.is_valid:
                return_dict['image_number'] = imgs[:image_num]
        return_dict['image_input'] = image_input
        ## image_attention_mask
        image_attention_mask = torch.zeros(self.image_len, dtype=torch.int64)
        image_attention_mask[0:image_num * self.per_image_len] = 1
        return_dict['image_attention_mask'] = image_attention_mask
        ## image_cls_mask
        image_cls_mask = torch.zeros(self.image_len, dtype=torch.int64)
        image_cls_mask[0:image_num * self.per_image_len:self.per_image_len] = 1
        return_dict['image_cls_mask'] = image_cls_mask
        
        ########### CLIP ###########
        ## clip_value
        if image_num > 0:
            title_clip = data_item_dict['title_clip']
            body_clip = data_item_dict['body_clip']
            if title_clip is None:
                clip_all = body_clip
            elif body_clip is None:
                clip_all = title_clip
            else:
                clip_all = np.concatenate([title_clip, body_clip], axis=1)
            clip_value = clip_all[:image_num, :sentence_num]
            return_dict['sentence_clip'] = torch.tensor(clip_value).type(torch.float)
            # test data
            summary_clip = data_item_dict['whole_summary_clip']
            if summary_clip is not None:
                return_dict['summary_clip'] = torch.tensor(summary_clip[:image_num]).type(torch.float).squeeze(-1)
            else:
                return_dict['summary_clip'] = torch.empty([image_num], dtype=torch.float)
        else:
            return_dict['sentence_clip'] = torch.empty([image_num, sentence_num], dtype=torch.float)
            return_dict['summary_clip'] = torch.empty([image_num], dtype=torch.float)

        ########### ENTITY ###########
        # if self.inline_entity:
        #     inline_entity_inputs_embeds, inline_entity_attention_mask, inline_scene_inputs_embeds, inline_scene_attention_mask = \
        #         self.sample_entity(data_item_dict, 'title_entity', 'body_entity', max_entity_num=self.max_inline_entity_num, max_scene_num=self.max_inline_scene_num)
        #     return_dict['inline_entity_inputs_embeds'] = inline_entity_inputs_embeds
        #     return_dict['inline_entity_attention_mask'] = inline_entity_attention_mask
        #     return_dict['inline_scene_inputs_embeds'] = inline_scene_inputs_embeds
        #     return_dict['inline_scene_attention_mask'] = inline_scene_attention_mask

        # entity_inputs_embeds, entity_attention_mask, scene_inputs_embeds, scene_attention_mask = \
        #     self.sample_entity(data_item_dict, 'title_entity', 'body_entity', max_entity_num=self.entity_num, max_scene_num=self.scene_num)
        # return_dict['entity_inputs_embeds'] = entity_inputs_embeds
        # return_dict['entity_attention_mask'] = entity_attention_mask
        # return_dict['scene_inputs_embeds'] = scene_inputs_embeds
        # return_dict['scene_attention_mask'] = scene_attention_mask

        entity_input_ids, entity_attention_mask = self.get_entity(data_item_dict, 'title_entity', 'body_entity', max_entity_len=self.max_entity_len)
        return_dict['entity_input_ids'] = entity_input_ids
        return_dict['entity_attention_mask'] = entity_attention_mask

        return return_dict
    


class MyModelDataset(Dataset):
    def __init__(self, dataset_config: DatasetConfig) -> None:
        super().__init__()
        self.file = dataset_config.new_src_file
        with open(self.file, 'rb') as fr:
            self.data_dict_list = pickle.load(fr)
    
    def __len__(self):
        return len(self.data_dict_list)

    def __getitem__(self, index):
        return self.data_dict_list[index]


def collate_fn(x):
    return_dict = {}
    for key in x[0].keys():
        if key in ['summary', 'sentence_clip', 'summary_clip', 'image_number', 'target_image_number']:
            return_dict[key] = [i[key] for i in x]
        else:
            return_dict[key] = torch.stack([i[key] for i in x], dim=0)
    return return_dict