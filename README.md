# Hazmat Label Identifier
The hazmat label identifier was designed to detect a defined range of hazmat labels in a range of environments. This was done through identifying a range of features each label posessed and return these to the user. It was not in the scope of this project to classify to the label itself. The range of labels the system needed to be able to classify various text, graphics and colours can be seen below: 

![Potential Labels](https://raw.githubusercontent.com/swithers19/Hazmat-Label-Identifier/master/test-images/PotentialLabels.JPG)

The development of determining what these attributes were or whether they existed at all required the use of OpenCV and basic python modules. The key functional requirements were the ability to segment and orientate a label from the environment as well as segment and classify the attributes such as the top and bottom colour of the label, class number which appears in the bottom corner, text and label identification.

An example output of the 5 key attributes can be seen below:
```sh
top: red
bottom: red
class: 3
text: COMBUSTIBLE
symbol: Flame
```
## The key design decisions include:
### 1. Segmenting the label from the background
This was achieved through a range of adaptive thresholding and edge detection processes given the strong edges the label possessed. This was then combined with some contour processes to ensure it matched properties that one would associate with the shape of the label.
### 2. Colour Determination
Colour recognition was fairly 'dumb' and used the HSL colour space with pre-defined colour ranges to determine how many pixels were in a given pre-defined colour range. 
### 3. Text Recognition
This process involved segmenting each character and then determing the histogram of oriented gradients of scaled shape which was then passed into an SVM where the expected character was output. This was done for each segmented character and a dictionary was used to look up words that may be expected. If their was a match between a dictionary word and the OCR's prediction, it was output as the text field.
### 4. Label Recognition
Label recognition was done in the same manner as text recognition, with some minor differences in the segmentation approach being somewhat simpler.
### 5. Class Number Detection
Similar to the text and label recognition, an SVM was used to classify a HOG descriptor The one difference was these characters were segmented using blobbing which returns a far cleaner outline. For this reason the resulting performance was signifigantly better.

## Results
The identifier was effective under a range of backgrounds and conditions, but struggled were lighting conditions were not optimal.
