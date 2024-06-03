import os

__all__ = ['DatasetConfig']


# configuration for my added parts
class DatasetConfig:
    def __init__(self, dataset: str, kg_dir: str) -> None:
        # pwd: EGMS
        self.dataset = dataset
        self.kg_dir = kg_dir
        self.data_dir = f'/path2EGMS/data/{self.dataset}'
        if not os.path.exists(self.data_dir):
            raise ValueError(f'Dataset \"{dataset}\" does not exist.') 
        self.img_dir = f'/path2MSMO/data/{self.dataset}/img'
        # self.src_file = f'{self.data_dir}/data_dict_with_clip.pkl'

        # self.new_src_file = f'{self.data_dir}/data_dict_ready2use.pkl'
        # self.new_src_file = f'{self.data_dir}/data_dict_ready2use_new.pkl'
        self.new_src_file = f'{self.data_dir}/data_dict_with_blip.pkl'

        # self.kg_emb_file = f'{self.data_dir}/../conceptnet/glove.transe.sgd.ent.npy'
        # self.kg_emb_file = f'{self.data_dir}/../conceptnet/tzw.ent-002.npy'
        
        # self.kg_ent2idx_file = f'{self.data_dir}/../conceptnet/ent2idx.txt'
        self.kg_ent2idx_file = os.path.join(self.kg_dir, 'entity2id.txt')
    