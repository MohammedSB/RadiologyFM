# Radiology Foundation Model.
Work in progress at MedARC.
# Run Experiments
To test the models for classification, you can use the following command.
```
PYTHONPATH=. python3 dinov2/run/eval/linear.py \
    --gpus <NUM_OF_GPUS> \
    --nodes <NUM_OF_NODES> \
    --batch-size <BATCH_SIZE> \
    --epochs <EPOCHS> \
    --val-epochs <VAL_EPOCHS> \
    --save-checkpoint-frequency <CHECKPOINT_EVERY> \
    --eval-period-epochs <EVAL_PERIOD> \
    --val-metric-type multilabel_auc \
    --finetune False
    --backbone dinov2
    --config-file <PATH_TO_DINOV2_FOLDER>/dinov2/configs/eval/vitb4_pretrain.yaml \
    --pretrained-weights <DINOV2_WEIGHTS_PATH> \
    --output-dir <OUTPUT_PATH> \
    --train-dataset CheXpert:split=TRAIN:root=<PATH_TO_DATASET>/CheXpert \
    --val-dataset CheXpert:split=VAL:root=<PATH_TO_DATASET>/CheXpert \
    --test-dataset CheXpert:split=TEST:root=<PATH_TO_DATASET>/CheXpert 
```
The above command will run a linear evaluation experiment with a DINOv2 ViT-B/14 model on the CheXpert dataset. The run will first search for the optimal hyperparameters by training the model with linear classifiers for `VAL_EPOCHS` number of epochs, testing on the validation set. After that, it will combine the validation and train set and train a new linear for `EPOCHS` number of epochs and evaluate it on the test set.

The parameter `--finetune` determines whether the backbone should be finetuned or not. An additional parameter `--backbone-learning-rate` determines the learning rate for tuning the backbone. The `--backbone` parameter determines the backbone to use, which is set to DINOv2 as default. Other options include `vit-large-imagenet21k`, `resnet-152-imagenet1k`, `vgg-19-imagenet1k`, `densenet-201-imagenet1k`, `msn-large-imagenet1k`, `mae-large-imagenet1k`, `clip-large`, `openclip-huge`, and `sam-large`.

The same command can be applied for segmentation evaluations, simply by changing the path from `dinov2/run/eval/linear.py` to `dinov2/run/eval/segmentation.py`. There are additional segmentation, including `--decoder` (linear or unet) and `--image-size`.
