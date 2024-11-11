# [Leveraging Entity Information for Cross-Modality Correlation Learning: The Entity-Guided Multimodal Summarization](https://aclanthology.org/2024.findings-acl.587.pdf)

> The rapid increase in multimedia data has spurred advancements in Multimodal Summarization with Multimodal Output (MSMO), which aims to produce a multimodal summary that integrates both text and relevant images. The inherent heterogeneity of content within multimodal inputs and outputs presents a significant challenge to the execution of MSMO. Traditional approaches typically adopt a holistic perspective on coarse image-text data or individual visual objects, overlooking the essential connections between objects and the entities they represent. To integrate the fine-grained entity knowledge, we propose an Entity-Guided Multimodal Summarization model (EGMS). Our model, building on BART, utilizes dual multimodal encoders with shared weights to process text-image and entity-image information concurrently. A gating mechanism then combines visual data for enhanced textual summary generation, while image selection is refined through knowledge distillation from a pre-trained vision-language model. Extensive experiments on public MSMO dataset validate the superiority of the EGMS method, which also prove the necessity to incorporate entity information into MSMO problem.

## Dataset
We use the Multimodal Summarization with Multimodal Output dataset MSMO in our experiments. MSMO can be downloaded from [MSMO](https://drive.google.com/drive/folders/1Wdq0I01SR84KfVjTum71fI8IceEybP-V).

## Code
* "configs" contains the config files for EGMS.
* "run.sh" is the bash script for training and testing EGMS.

### Dependencies
* Python 3
* torch=1.12.1
* transformers=4.23.1
* Numpy

### Running

To run EGMS, simply run:
```
run.sh
```
> available parameters can be seen in run.sh

## Citation
If you use our model or code, please kindly cite it as follows:      
```
@inproceedings{zhang2024leveraging,
  title={Leveraging Entity Information for Cross-Modality Correlation Learning: The Entity-Guided Multimodal Summarization},
  author={Zhang, Yanghai and Liu, Ye and Wu, Shiwei and Zhang, Kai and Liu, Xukai and Liu, Qi and Chen, Enhong},
  booktitle={Findings of the Association for Computational Linguistics ACL 2024},
  pages={9851--9862},
  year={2024}
}
```
