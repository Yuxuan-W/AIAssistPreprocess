AQVSR Dataset Pre-Processing
=====
Dataset Pre-Processing part of AQVSR dataset, which is proposed in

[AssistSR: Affordance-centric Question-driven Video Segment Retrieval](https://arxiv.org/abs/2111.15050)

## Run
To run this pre-processing code, you need to install Detectron2 at 
https://github.com/facebookresearch/grid-feats-vqa  
Or run the command below: 
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@ffff8ac'
```

Then download R-50 backbone from the website below:
https://github.com/vedanuj/grid-feats-vqa and then put the R-50.th file into a new folder named ./checkpoint.

Then run the main.py file using the command line below:
```
python main.py
```
Or run the methods included in main.py separately.


## Acknowledgement

This code borrowed components from the following projects: 
[transformers](https://github.com/huggingface/transformers),
[TVRetrieval](https://github.com/jayleicn/TVRetrieval),
we thank the authors for open-sourcing these great projects!
