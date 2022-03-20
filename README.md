# Face Recognition using Siamese Network

### Dataset - http://vis-www.cs.umass.edu/lfw/

### Paper - https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

### Youtube Video Reffered for help [Video By Nicolas](https://www.youtube.com/watch?v=bK_k7eebGgc&list=PLgNJO2hghbmhHuhURAGbe6KWpiYZt0AMH&index=1)

## How to Use and Remarks

- I have used used Docker for the hassle free setup of the tensorflow gpu with jupyter notebook
- All the training, preprocessing and inferences are done with this.

### First Create the Following folder structure

```bash
data/
├── anchor/
├── positive/
└── negative/
```

#### and

```bash
application_data/
├── input_image/
└── verification_images/
```

### Fire up your docker container

`docker compose up`

1.  > You can run the notebook in that environment but to add the positive and anchor images you have to use `imageCollector,py` file on
    > your local machine due to the fact that docker doesn't connect to hardware directly. If you are a linux user you can add
    > "/dev/video0:/dev/video0" in the devices in docker compose file. Otherwise its much of a hassle to set it up.

2.  > Download the data,unzip it and move the images to desired folders using the notebook.
3.  > Go through the notebook for all other steps

#### and for verification it goes the same

> Run `verification.py`
> I have configured it to run on CPU, you can change to GPU as well.
