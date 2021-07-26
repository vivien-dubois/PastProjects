Deep Learning Based Image Quality Assessment Metrics

This package contains two parts: the first is contained in the IQA folder
and consists of a script to compute quality estimates using the models
described in my report and the second is contained in the autoencoder
folder and consists of a Jupyter notebook that I used on Google Colab to
train autoencoders using the IQA metrics of the first part as loss
functions.

== Part 1: IQA ==
This folder contains a script called IQA.py that computes full reference
quality estimates for given images. Detailed usage informations are
available through the -h or --help options. It has been created using
Tensorflow 1.14.0 and will probably work with any later releases.
The script provides the baseline WaDIQaM model, the multiscale MS-WaDIQaM
model and the aggregated model decribed in the report.

A Jupyter notebook that demonstrates the usage of the training code is
also provided. In the beginning of the notebook variables indicate the
location of the image databases. The expected file structure is as
follows:

TID2013
<tid_loc>/
    distorted_images
    reference_images
    mos.txt
    
LIVE
<live_loc>/
    fastfading
    gblur
    jp2k
    refimgs
    wn
    dmos.mat
    refnames_all.mat
    
CSIQ
<csiq_loc>/
    distorted_images
    reference_images
    csiq.DMOS.xlsx
    
JPEG_AI and JPEG_XL
<hdai_loc>/
    Distorted
        "Contains the distorted images of both XL and AI"
    References
        "Contains the reference images of both XL and AI"
    AIscores.csv
    names_scores_XL.csv


== Part 2: autoencoder ==
During the project many models and images were created totaling several
gigabytes of data. They are not included here.

This folder contains a Juypter notebook and the utilities needed to run
it on Google Colab. To use it on this platform add the contents of this
folder to a Google Drive as follows:

<Drive Root>/
    tid2013/
        distorted_images
        reference_images
        mos.txt
    Colab Notebooks/
        autoencoder.ipynb
        utils.py
        adversarial.py
        
You can then run the notebook on Colab and it should work without issues.
It is possible that some folders are not created automatically and that
some lines of code will throw exceptions sayings that some paths do not
exist when saving data. Creating an empty folder with the correct name
will fix it.

If you only want to look at the contents of the notebook any jupyter
install will do but be aware that many paths may be broken if you try to
execute it outside Colab.

This code was written for Tensorflow 1.15.0, it will probably work with
later releases but will not work for any previous release.
