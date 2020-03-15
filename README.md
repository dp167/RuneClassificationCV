# RuneClassificationCV
Using Computer Vision to build a image classifier that can classify Rune tiles, for CS410p At PSU

MVP:

for this project includes the ability to generate a data set from smartphone photos taken of runes and be able to classify 6 specific runes via a file of testing images with an accuracy of at least 70%.

Reach goals:
  1. produce bounding boxes around particular runes in an image of multiple runes with each rune labled correctly
  
  2. Deploy mobile application to allow for real time classification of runes including the applicatio of goal 1.
  
  2.a. add features to estimate rune distance and output the possible meaning of the rune formations
  
Our initial goal is to have a compile time program that can identify Runes in non- preprocessed testing images. Our reach goals involve deploying a runtime solution that will be able to identify Runes in an image or video feed from a mobile application. 

Progress and milestones will be documented in the projects README.md

Steps to implement solution:
Build image cropping program, test it with self taken photos
Build Image preprocessing program
Build CNN Program and train it on the generated data from the preprocessing program

MILESTONES AND DELIVERABLES :
   
      02/06- Project Proposal due
      02/20- Working Code review, revise plan and document progress
      02/27- Code review, Determine whether to pursue reach goals (if  1,2,3 are done)
      03/05- Work on bugs, refine final deliverable
      03/10- Presentation of Deliverables due
      03/15 Final Submission due

MILESTONE UPDATE 02/20/2020
  ----see full details at https://github.com/dp167/RuneClassificationCV/blob/master/milestoneupdate_codereview02_20.pdf

Short term goals before next milestone (02/27) to stay on track:
      
      - Perfect autocroping program
      - Have working image processing program and generated minimum 50 images per Ruin for a testing data set
      - Test CNN program on the testing data
              -Revise our approach to data augmentation based on the results here

MILESTONE UPDATE 02/27/2020
------- see full details at:
https://github.com/dp167/RuneClassificationCV/blob/master/milestoneupdate_codereview02_27.pdf

Short term goals before next milestone (03/04) to stay on track:
      
        - Deploy neural network to classify images in an android application. 
        - Finalized image preprocessing program.
