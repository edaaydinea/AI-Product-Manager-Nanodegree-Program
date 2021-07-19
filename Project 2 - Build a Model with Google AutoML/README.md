# Build a Model with Google AutoML

# Overview

In this project, we need to build a labeled dataset that distinguishes between healthy and pneumonia x-ray images.This dataset would eventually be used to build a 
product that helps doctors quickly identify cases of pneumonia in children.we are going to take the next step and build the classification model that would serve as the 
backbone of this product.For this task,we will use Google AutoML,an automated machine learning tool that will allow us to build the model and host it in the cloud. 
In order to appreciate how training data impact models,we will build models with 4 variants of the dataset.The goals of this product are 
1. Help flag serious cases,
2. Quickly identify healthy cases,
3. And, generally, act as a diagnostic aid for doctors.

# The Four Parts of the Project

We will train four different models using four variants of the pneumonia dataset.The dataset contains childrens' chest x-ray images, 
and that they are classified into two classes, "normal" and "pneumonia". The following cases describe the steps we must take to create each model.

## Case 1: Create a binary classifier to detect pneumonia using chest x-rays

We'll start by training a model simply using 100 images from the “normal” class and 100 images from the “pneumonia” class.We are asked to evaluate the following:

- Train/test split: How much data is used for training and how much is used for testing?
- Confusion matrix: What do each of the cells in the confusion matrix describe?
- Precision & recall: What do these measure and how are they calculated?

## Case 2: Create an unbalanced binary classifier

Next, use 100 images from the “normal” class, and add 200 more "pneumonia" class images. At this moment, the total count of images must be:

“normal” class = 100
"pneumonia" class = 300
The model will be trained on very unbalanced classes; this will show what happens when the number of images in different classes is very different. 
 We are asked to evaluate:

- Confusion matrix: What does the confusion matrix tell you about unbalanced data?
- Precision & recall: How are precision and recall affected?
- Unbalanced classes: How do unbalanced classes impact a machine learning model?

## Case 3: Create a binary classifier with dirty data

In this iteration, start with the original dataset of 100 "normal" and 100 "pneumonia" images. Then switch the labels of 30 images in each class. 
After we've done this, 30% of the data are mislabeled. We are asked to evaluate:

- Confusion matrix: How is the confusion matrix affected?
- Precision & recall: How are precision and recall affected?
- Dirty data: How do dirty data impact the model?

## Case 4: Create a three-class model with the classes “normal”, “bacterial pneumonia”, and “viral pneumonia”

For the final model, note that the "pneumonia" images actually have two different classes: "bacterial" pneumonia and "viral" pneumonia. 
These labels are indicated in the image filenames. For this model, add 100 "normal" images, 100 "bacterial pneumonia" images, and 100 "viral pneumonia" images (for a total of 3 classes). 
We are asked to evaluate:

- Confusion matrix: What does the 3-class confusion matrix look like?
- Precision & recall: What are the model's precision and recall? How are these measures calculated?
- More data: Can we continue to add data to each class (while keeping the data balanced) and get the model to 85% precision and recall? How many data did you end up using?

# Google Cloud AutoML Vision
The following tutorial will guide you through the process of creating and training a model on Google Cloud's AutoML Vision platform. Go through the following steps to complete the project.

Steps:

1. Download Kaggle Chest X-Ray Images (Pneumonia) Dataset, if you haven't already: (https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia#chest_xray.zip).
2. Go to Google AutoML: (https://cloud.google.com/automl/).
3. Click “TRY AUTOML” and select “AutoML Vision”.
4. Sign in to Google Developer or create an account.
5. After you log in, head to the AutoML Vision Console: (https://console.cloud.google.com/vision) and Accept the terms of service.
6. Next create a project with any title.
Here is a link to the ![chest-xray.zip](https://drive.google.com/file/d/1maDw5RxvyFXiQTVc2lSLCYdWo6Z8FKxG/view?usp=sharing) in case you are unable to download it from Kaggle.

![image1](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%202%20-%20Build%20a%20Model%20with%20Google%20AutoML/Screenhsots/image1.png)
![create-project](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%202%20-%20Build%20a%20Model%20with%20Google%20AutoML/Screenhsots/create-project.png)

7. Once on the dashboard click “Get Started” on AutoML Image Classification.
8. Accept all the screens then you will reach the last page where you have to set up billing and Vision API.

![gcscreenshot3](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%202%20-%20Build%20a%20Model%20with%20Google%20AutoML/Screenhsots/gcscreenshot3.png)

9. When you go to billing you should see an option to activate free $300 credit at the top of the screen. Click “ACTIVATE”.

![gcscreenshot4](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%202%20-%20Build%20a%20Model%20with%20Google%20AutoML/Screenhsots/gcscreenshot4.png)

10. Select “Individual” account type and input all necessary info including card. Google does not charge this card unless you manually upgrade to a paid account.

![gcscreenshot5](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%202%20-%20Build%20a%20Model%20with%20Google%20AutoML/Screenhsots/gcscreenshot5.png)

### We strongly advise students to not upgrade to a PAID account.
Your credit card will be charged in case you switch to a PAID account. This is done through the Google Cloud Console. Switching to a PAID account requires clicking the Activate button.

Please bear in mind the following guidelines from the ![Google Cloud Platform Instructions](https://support.google.com/cloud/answer/7006543?hl=en_)

1. The free trial ends when you use all of your credit, or after 3 months, whichever happens first.

2. **You may also lose your data after the trial period is over.**

We recommend that you review the full set of guidelines from Google Cloud Platform services.

**Set up Budgets and Alerts**
You can use the Budget Alert tool on GCP to set up alerts when you have used 50%, 75% and 90% of your free credits within your trial period. This way you will be alerted as you are approaching the credit limit with your trial period.

Here is a video provided by Google to help you set up budget alerts:

Video detailing how to ![create Budgets and Alerts](https://www.youtube.com/watch?v=F4omjjMZ54k&t=132s)
![question](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%202%20-%20Build%20a%20Model%20with%20Google%20AutoML/Screenhsots/question.png)

Yes, I understand I will be charged if I switch to a paid account, and I will be responsible for any charges.

11. Once you have completed the billing you can go back to the project set up page and click “SET UP NOW” to enable the vision API. This will automatically set up the API and should redirect you select the Google Cloud Project.
12. In the dropdown you should see the project that you created. Select the project and continue.

![gcscreenshot6](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%202%20-%20Build%20a%20Model%20with%20Google%20AutoML/Screenhsots/gcscreenshot6.png)

13. Next, you should see the Datasets page. Click “NEW DATASET” at the top of the page.

![gcscreenshot7](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%202%20-%20Build%20a%20Model%20with%20Google%20AutoML/Screenhsots/gcscreenshot7.png)

14. Select “Single-Label Classification”.

![gcscreenshot8](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%202%20-%20Build%20a%20Model%20with%20Google%20AutoML/Screenhsots/gcscreenshot8.png)

15. Create your dataset.
    1. Give the dataset a name.
    2. Open the folder of Chest Xrays you downloaded from Kaggle and go into the train folder.
    3. In a separate folder create a directory with the following structure:
         - AutoMLDataset/
         - normal/
         - pneumonia/
    4. Then copy between 100-300 images of each type of each classification from the original train/ folder (normal, pneumonia) into the appropriate new folder. The exact number of images in each folder will depend on which of the four parts of the project you are solving, as described in the previous concept: Project Overview. Once you have your images in the new folders then zip the whole **AutoMLDataset/** folder to create **automldataset.zip**.

![gcscreenshot9](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%202%20-%20Build%20a%20Model%20with%20Google%20AutoML/Screenhsots/gcscreenshot9.png)

16. Now back in the AutoML Console Dataset go ahead and upload the zip file with all the images. Again, make sure you zipped the top-level folder that contains two folders: **normal/** and **pneumonia/** with the associated sets of 100-300 images in each folder (the exact number of images will depend on which of the four parts of the project you are solving as described in the previous concept: Project Overview). This will allow AutoML to automatically detect the correct ground truth label for each image by looking at the name of the folder in which it is contained. Once the zip file is attached go ahead and “CREATE DATASET”. For our current labeling scheme, there is no need to select the button to "Enable multi-label classification" (this allows you to apply multiple labels to a single image). This step make take a few minutes to upload all the images and set up the dataset.

17. Once the data is ready you should see a screen with all the images you added as well as a label of “normal” or “pneumonia” under each image. These are the images and ground truth labels that will be used to train/test/validate your pneumonia detector model.

![gcscreenshot10](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%202%20-%20Build%20a%20Model%20with%20Google%20AutoML/Screenhsots/gcscreenshot10.png)

18. Next click on the “TRAIN” tab and here you will see the distribution of images as well as the splits for training, validation, and test images.

![gcscreenshot11](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%202%20-%20Build%20a%20Model%20with%20Google%20AutoML/Screenhsots/gcscreenshot11.png)

19. Click “START TRAINING” to begin training your pneumonia classification model. In the options keep all the defaults (Cloud-hosted, 8 node hour) and then start the training.

![model-training](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%202%20-%20Build%20a%20Model%20with%20Google%20AutoML/Screenhsots/model-training.png)

20. Now the model is in the training phase. Go grab a cup of coffee and stretch your legs. Once the model training is done you will be notified via email.

21. Once the training is complete you can click on the “EVALUATE” tab and you will see various metrics about the model such as precision, recall, and a confusion matrix. Try to understand what each of these metrics means and how they are calculated.

![model-results](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%202%20-%20Build%20a%20Model%20with%20Google%20AutoML/Screenhsots/model-results.png)

22. Optional: Next click on the “TEST&USE” tab and this is where you can test the model on new data that the model has never seen. If you click “UPLOAD IMAGES” and select an image from the Kaggle dataset that you did NOT use in the training set of images and watch the model predict whether or not the patient in the x-ray has pneumonia or not. Try to find images that produce both correct and wrong predictions.

![gcscreenshot12](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%202%20-%20Build%20a%20Model%20with%20Google%20AutoML/Screenhsots/gcscreenshot12.png)

23. That's it! You've built your first AI model! Now that you know how to upload data and train the model, follow the steps above for 3 additional variations on the dataset, for a total of 4 models. We recommend creating new folders, new zip files, and new datasets in AutoML so that you can keep track of which iteration you're on and retain the results of each trained model for reference:
    1. Clean/Unbalanced: Create a model using 100 "normal" images and 300 "pneumonia" images.
    2. Dirty/Balanced: Create a model with 100 "normal" images and 100 "pneumonia" images, but deliberately create a "dirty" dataset by putting 30 "normal" images in the pneumonia folder and 30 "pneumonia" images in the normal folder.
    3. 3-Class: Note that the filenames of images in the "pneumonia" class have 2 standards; for example: **person75_bacteria_366.jpeg** and **person80_virus_150.jpeg**. The inclusion of the word "bacteria" indicates that this was a case of bacterial pneumonia, and the word "virus" means this was a case of viral pneumonia. Go back and create a dataset with 3 classes: "normal", "bacterial pneumonia" and "viral pneumonia".
