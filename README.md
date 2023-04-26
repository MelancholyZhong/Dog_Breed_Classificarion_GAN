# Dog-Breeds-Classification-Generation

In this project, we explored several topics in deep network using the Stanford Dogs Dataset. Firstly, we implemented a dog breed classifier. By implementing and comparing 5 different architectures, we found out that the ResNet50 architecture has the best performance in this task and reached a 57% accuracy for dog breed classification. In addition, we were interested in the image generator. Therefore, we also extended our project to a dog breed generator using the conditional deep convolution generative adversarial network(cDCGAN).

If encounter any problems, please feel free to contact us via email hu.hui1@northeastern.edu or zhong.yao@northeastern.edu

## Environment

Development: Mac OS M1 chip and VSCode as IDE

Train Models: Google Colab GPU

## Links

- link to video presentation: https://youtu.be/BPHOXmlmpD4

- link to ppt: https://docs.google.com/presentation/d/1YrftRBZ3CTmKUfN8IpXcZzoyy9kvLE-T68WADbDohSI/edit?usp=sharing

- link to github: https://github.com/Hui-Hwoo/Dog-Breeds-Classification-Generation

- link to trained models:

## How to run the code

For each file, click run in the IDE or run the python file in terminal

### Preprocess part

Run `python image_preprocess.py` in terminal.

Make sure to have a folder named `stanford-dogs-dataset` to save data of Annotation, Images and lists downloaded from http://vision.stanford.edu/aditya86/ImageNetDogs/

### Classification part

Run `python image_classification.py` in terminal.
