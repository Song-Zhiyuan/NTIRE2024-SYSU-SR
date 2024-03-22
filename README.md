# [NTIRE 2024 Challenge on Image Super-Resolution (x4)](https://cvlai.net/ntire/2024/) @ [CVPR 2024](https://cvpr.thecvf.com/)

## How to get factsheet?

The compiled pdf file together with a zip with .tex factsheet source files are located at
[`factsheet`](./factsheet)

## How to test the baseline model?

1. `git clone https://github.com/Song-Zhiyuan/NTIRE2024-SYSU-SR.git`
2. Select the model you would like to test from [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 14
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.
    - We provide MODEL (team14): SYSU-SR(HGD) The code and pretrained models of the three models are provided. Switch models through commenting the code in [test_demo.py](./test_demo.py#L19). Three baselines are all test normally with `run.sh`.

## How to add your model to this baseline?
1. Our team ID is 14.
2. Put your the code of your model in `./models/[Your_Team_ID]_[Your_Model_Name].py`
   - Please add **only one** file in the folder `./models`. **Please do not add other submodules**.
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02 
3. Put the pretrained model in `./model_zoo/[Your_Team_ID]_[Your_Model_Name].[pth or pt or ckpt]`
   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02
   - Note:  Please provide a download link for the pretrained model, if the file size exceeds **100 MB**. Put the link in `./model_zoo/[Your_Team_ID]_[Your_Model_Name].txt`: e.g. [team14_HGD.txt](model_zoo/team14_HGD.txt)
4. Add your model to the model loader `./test_demo/select_model` as follows:
    ```python
        elif model_id == [Your_Team_ID]:
            # define your model and load the checkpoint
    ```
   - Note: Please set the correct data_range, either 255.0 or 1.0
5. Send us the command to download your code, e.g, 
   - `git clone [Your repository link]`
   - We will do the following steps to add your code and model checkpoint to the repository.
   
## How to calculate the number of parameters, FLOPs

```python
    from utils.model_summary import get_model_flops
    from models.team14_HGD import EnsembleModel
    model = EnsembleModel()
    
    input_dim = (3, 256, 256)  # set the input dimension
    flops = get_model_flops(model, input_dim, False)
    flops = flops / 10 ** 9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
```

## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 
