

# Leveraging Vulnerabilities Towards Image Compression for Robust Classification of Natural and Generated Images

Manjary P. Gangan<sup>a</sup>, Anoop Kadan<sup>b</sup>, and Lajish V. L.<sup>a</sup> </br>
<sup>a</sup> Department of Computer Science, University of Calicut, India </br>
<sup>b</sup> University of Southampton, United Kingdom


:memo: Paper : [https://arxiv.org/abs/2308.07279](https://arxiv.org/abs/2308.07279)</br>
:earth_asia: Link: [https://dcs.uoc.ac.in/cida/projects/dif/mcevit.html](https://dcs.uoc.ac.in/cida/projects/dif/mcevit.html) 


**Abstract**: The works in literature classifying natural and computer generated images are mostly designed as binary tasks either considering <i>natural images versus computer graphics images</i> only or <i>natural images versus GAN generated images</i> only, but not natural images versus both classes of the generated images. Also, even though this forensic classification task of distinguishing natural and computer generated images gets the support of the new convolutional neural networks and transformer based architectures that can give remarkable classification accuracies, they are seen to fail over the images that have undergone some post-processing operations usually performed to deceive the forensic algorithms, such as JPEG compression, gaussian noise, etc. This work proposes a robust approach towards distinguishing natural and computer generated images including both, computer graphics and GAN generated images using a fusion of two vision transformers where each of the transformer networks operates in different color spaces, one in RGB and the other in YCbCr color space. The proposed approach achieves high performance gain when compared to a set of baselines, and also achieves higher robustness and generalizability than the baselines. The features of the proposed model when visualized are seen to obtain higher separability for the classes than the input image features and the baseline features. This work also studies the attention map visualizations of the networks of the fused model and observes that the proposed methodology can capture more image information relevant to the forensic task of classifying natural and generated images. 


***For inquiries, please contact:*** </br>
&nbsp; Manjary P. Gangan, University of Calicut, Kerala, India. :email: manjaryp@gmail.com :earth_asia: [website](https://dcs.uoc.ac.in/~manjary/) </br>
&nbsp; Anoop Kadan, University of Southampton, UK. :email: anoopkadan23@gmail.com :earth_asia: [website](https://www.southampton.ac.uk/people/65qvt5/doctor-anoop-kadan)</br>


## Citation
```
@article{gangan2023robust,
      title={Leveraging Vulnerabilities Towards Image Compression for Robust Classification of Natural and Generated Images}, 
      author={{Manjary P. Gangan} and {Anoop Kadan} and {Lajish V L}},
      year={2023},
      eprint={2308.07279},
      archivePrefix={arXiv},
      doi={https://doi.org/10.48550/arXiv.2308.07279}
}
```

## Acknowledgement
This work was supported by the Women Scientist Scheme-A (WOS-A) for Research in Basic/Applied Science from the Department of Science and Technology (DST) of the Government of India 




