# Create a Medical Image Annotation Job

# Introduction

Our goal, as a product owner is to build a product that helps doctors quickly identify cases of pneumonia in children. You'll want to build a classification system that
1. Can help flag serious cases
2. Quickly identify healthy cases
3. And, generally, act as a diagnostic aid for doctors As such, this project is designed to test our ability to build a labeled dataset that distinguishes between healthy and pneumonia x-ray images.This can be used by ML engineers later on down the line to build a classification product.Our main task will be to create a data labeling job using [Appen's platform](https://client.appen.com/jobs). If we have not yet, we need to create an Appen account and login to complete this project.

If we intend to purchase any services through Appen, we need want to use a valid business email.so, we can receive notifications. But for the project, if we do not have a business email, we can use the above workaround in the meantime.

we will not be required to actually launch the data labeling job, we be evaluated on submitted documents and design only by Udacity mentor team.

# The Data

The dataset we'll be given is a modified version of this [Kaggle chest x-ray dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia), with most labels removed. Every piece of data is a chest x-ray image. We may see images taken that are slightly different in size and taken under slightly different exposure times. A typical, labeled image is shown below.

![Annotated_Chest_Xray](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%201-%20Create%20a%20Medical%20Image%20Annotation/Screenshots/annotated-chest-xray.png)
<p align="center"> A labeled, healthy, chest x-ray image. Pay close attention to the two lungs and diaphragm (below the lungs) </p>

# What Does Pneumonia Look Like?

This is a challenging task because it is not always clear when pneumonia symptoms are present or not in an image. As such, your system is not meant to be a replacement for a doctor, only to aid in quickly identifying healthy patients and surfacing potential cases of pneumonia.

we should design a data annotation job, such that a non-expert can identify more noticeable cases of pneumonia. Since we are designing for a non-expert annotator, you should design for failure; this means including some way to capture uncertainty in your data labels and test questions.

So, what indicates pneumonia and what kind of advice and examples can we give potential annotators?

There are a few different visual symptoms that indicate pneumonia. The most important areas to have annotators pay attention to are the lungs and the diaphragm.
1. A normal, healthy image will depict clear lungs without any areas of abnormal cloudiness/opacity; there may be structured, web-like vasculature in the lungs but otherwise that area should be clear. In healthy images, you are also more likely to see a diaphragm shadow.
2. A pneumonia image may include a few things: areas of cloudiness/opacity in several concentrated areas or one large area. You may also see a general pattern of opacity that obscures the structure of the lungs, heart and diaphragm.

![Healthy_Example](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%201-%20Create%20a%20Medical%20Image%20Annotation/Screenshots/healthy-example.png)
         <p align="center"> some characteristics of a healthy image: a clear lung area </p>
 
![Pneumonia_Examples](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%201-%20Create%20a%20Medical%20Image%20Annotation/Screenshots/pneumonia-examples.png)
            <p align="center"> Examples of pneumonia symptoms: (Left) a concentrated, opaque area in the lungs, (Right) multiple, smaller opaque areas throughout the lung       area and any diaphragm shadow is obscured. </p>


# Create a Medical Image Annotation Job with Appen 

## Step 1 : Data
![Data](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%201-%20Create%20a%20Medical%20Image%20Annotation/Screenshots/Data.png)

## Step 2: Desing Your Job
![Design Your Job](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%201-%20Create%20a%20Medical%20Image%20Annotation/Screenshots/Design%20your%20job.png)

## Step 3: Quality
![Negative_Answers](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%201-%20Create%20a%20Medical%20Image%20Annotation/Screenshots/Test_Questions_Negative.png)
![Positive_Answers](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%201-%20Create%20a%20Medical%20Image%20Annotation/Screenshots/Test_Questions_Positive.png)

## Step 4: Distribution
![Distribution](https://github.com/edaaydinea/AI-Product-Manager-Nanodegree-Program/blob/main/Project%201-%20Create%20a%20Medical%20Image%20Annotation/Screenshots/Distribution.png)
