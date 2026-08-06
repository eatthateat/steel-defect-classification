## Goal
The goal of the project was to classify six kinds of defects of the hot-rolled steel strip viz. rolled-in scale (RS), patches (Pa), crazing (Cr), pitted surface (PS), inclusion (In) and scratches (Sc).

## Approach
Used **NEU Surface Defect Database** as a [Dataset](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database).

The approach used is to take a **ResNet18** pretrained model, fine-tune fourth sequential and fully connected one layers for our specific task.

## Metrics (to be fixed)
Current gathered metrics are not to be displayed due to known and unknown bugs. Currently in process of validation accuracy goes all the way up to 100%, which is not healthy behavior signalizing of some kind of leak. Also theres no test data split now.

## Quick start

1. Install the requirements.
```sh
pip install -r requirements.txt
```

2. Download dataset in a desirable path. (requires kaggle account authorization)
```sh
#!/bin/bash
curl -L -o ~/Downloads/neu-surface-defect-database.zip\
  https://www.kaggle.com/api/v1/datasets/download/kaustubhdikshit/neu-surface-defect-database
```

3. Cofigure the `config/baseline.yaml` to your liking (especially the dataset path).

4. Start training loop.
```sh
python -m src.train
```

## Known bugs (to be fixed)
1. transforms.Resize((200, 200)) – The NEU dataset is already 200x200, but ResNet expects 224x224.
2. class_to_idx silently returns None for unknown classes.
3. transforms.Normalize is commented out in both transformations.
4. `imread` in the dataset silently returns `None` for a null path; an exception should be raised.
5. There is an issue with the parameters in “`image = cv2.imread(self.image_paths[idx], cv2.COLOR_BGR2RGB)`”; the conversion does not actually take place.


