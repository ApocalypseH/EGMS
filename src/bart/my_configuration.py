import torch
import os
import json

__all__ = ['MyConfig']


# configuration for my added parts
class MyConfig:
    def __init__(self, config_file = None, save_dir = None, device = 'cpu', num_entity = None, d_entity = None) -> None:
        self.config_file = config_file
        if config_file is not None:
            with open(config_file, 'r') as fr:
                config_dict = json.load(fr)

        self.d_model = 1024 if config_file is None else config_dict['d_model']
        self.bart_model = 'facebook/bart-large-cnn' if config_file is None else config_dict['bart_model']
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.batch_size = 2 if config_file is None else config_dict['batch_size']
        self.epoch_num = 1 if config_file is None else config_dict['epoch_num']
        # lr
        self.learning_rate = 1e-6 if config_file is None else config_dict['learning_rate']
        self.start_factor = 1.0 if config_file is None else config_dict['start_factor']
        self.end_factor = 1.0 if config_file is None else config_dict['end_factor']
        self.warm_up_step_num = 200 if config_file is None else config_dict['warm_up_step_num']

        self.save_step = 2000 if config_file is None else config_dict['save_step']
        self.log_step = 100 if config_file is None else config_dict['log_step']
        self.save_dir = '/path2EGMS/saved_model/work_dir' if save_dir is None else save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Text
        # self.decoder_attention_heads = None
        self.self_attention_heads = 16 if config_file is None else config_dict['self_attention_heads']
        self.cross_attention_heads = 16 if config_file is None else config_dict['cross_attention_heads']
        self.attention_dropout = 0.0 if config_file is None else config_dict['attention_dropout']
        self.dropout = 0.1 if config_file is None else config_dict['dropout']
        self.activation_function = 'gelu' if config_file is None else config_dict['activation_function']
        self.activation_dropout = 0.0 if config_file is None else config_dict['activation_dropout']
        self.decoder_ffn_dim = 4096 if config_file is None else config_dict['decoder_ffn_dim']

        self.encoder_layers = 1 if config_file is None else config_dict['encoder_layers'] # kg encoder
        self.decoder_layerdrop = 0.0 if config_file is None else config_dict['decoder_layerdrop'] # 忽略掉某一层的概率
        self.scale_embedding = False if config_file is None else config_dict['scale_embedding']

        # kg
        # self.d_entity = 100 if config_file is None else config_dict['d_entity']
        if config_dict.get('num_entity') is None and num_entity is None:
            raise ValueError('You have to specifiy the num entity.')
        else:
            self.num_entity = num_entity if num_entity is not None else config_dict.get('num_entity')

        if config_dict.get('d_entity') is None and d_entity is None:
            raise ValueError('You have to specifiy the d entity.')
        else:
            self.d_entity = d_entity if d_entity is not None else config_dict.get('d_entity')    
        
        self.entity_pad_token_id = 0
        self.scene_token_id = 1
        self.entity_id_offset = 2
        self.sent_ent_gate_hidden_dim = 512

        self.max_entity_num = 64 if config_file is None else config_dict['max_entity_num']
        self.max_scene_num = 32 if config_file is None else config_dict['max_scene_num']
        self.entity_partial = 0.5 if config_file is None else config_dict['entity_partial']
        self.scene_partial = 0.5 if config_file is None else config_dict['scene_partial']

        # Vision
        # self.image_size = 224 if config_file is None else config_dict['image_size']
        # self.patch_size = 32 if config_file is None else config_dict['patch_size'] # 7*7 grid
        self.max_image_num = 8 if config_file is None else config_dict['max_image_num']
        self.blip_query_num = 32 if config_file is None else config_dict['blip_query_num']
        self.d_image = 768 if config_file is None else config_dict['d_image']

        # KL DIV
        self.kl_hidden_dim = 1024 if config_file is None else config_dict['kl_hidden_dim']
        self.temperature = 10 if config_file is None else config_dict['temperature'] # softmax temperature

        # Loss
        self.use_kl_div = True if config_file is None else config_dict['use_kl_div']
        self.kl_div_weight = 1 if config_file is None else config_dict['kl_div_weight']
        self.lm_loss_weight = 1 if config_file is None else config_dict['lm_loss_weight']

        # Encoder Combination
        self.encoder_combination = False if config_file is None else config_dict['encoder_combination']
        self.kg_encoder_weight = 1 if config_file is None else config_dict['kg_encoder_weight']

        # Inline Entity
        self.inline_entity = False if config_file is None else config_dict['inline_entity']
        self.max_inline_entity_num = 16 if config_file is None else config_dict['max_inline_entity_num']
        self.max_inline_scene_num = 8 if config_file is None else config_dict['max_inline_scene_num']

        # Dual Encoder
        self.share_weights = False

        # Generation
        self.min_length = 56 if config_file is None else config_dict['min_length']
        self.max_length = 142 if config_file is None else config_dict['max_length']
        self.length_penalty = 2.0 if config_file is None else config_dict['length_penalty']

        ####### model option
        self.has_image_select = True if config_file is None else config_dict['has_image_select']
        self.has_text_image = True if config_file is None else config_dict['has_text_image']
        self.has_entity_image = True if config_file is None else config_dict['has_entity_image']

        self.is_loss_weight = 1. if config_file is None else config_dict['is_loss_weight']
    