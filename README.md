# ECN - 316 Course Project

## Group Members

- Ankit Kataria - 1611606
- Deepesh Pathak - 16116018
- Sai Himal Allu - 16116055

## Image Denoising Pipeline

Image denoising is a research problem that has attracted significant amounts of attention from the academic community. Images often suffer from corruptions which can be categorized into two categories: blur and noise.
Noise is a factor that mainly appears during the acquisition, transmission, and retrieval of the signals. The purpose of any denoising algorithm is to remove such noise while maintaining the maximum amount of details in the image.

Denoising of images refers to the process of removing noise from a signal.
Historically and even now, when images are stored using photographic film and magnetic tape, noise was introduced due to the grain structure of the medium. 

Salt and pepper noise is one such typical example of noise in images where the pixels in the image are very different in colour or intensity compared to a reference pixel. The noisy pixels in an image in this case bear no relation to the properties (colour, intensity) of the surrounding pixels. These are usually seen as dark and white dots in the image, hence the term salt and pepper noise. In most cases, this noise affects only a small amount of pixels in the image. 
In academic literature, noise is usually formulated with the help of a Gaussian model. From a mathematical point of view, this follows directly from the Central Limit Theorem, since different noises adding together would tend to approach a Gaussian distribution. In either case, the noise at different pixels can be either correlated or uncorrelated; in many cases, noise values at different pixels are modelled as being independent and identically distributed, and hence uncorrelated.

In this project, we tried experimenting with different standard approaches to this problem which are detailed in the forthcoming sections with the singular aim of creating an efficient pipeline for the purpose of denoising.

For the finer details of this project, please refer to the [project report][2] or the [slides][1] used for presenting this project 

```bash
$ virtualenv venv

$ source venv/bin/activate

$ pip install -r requirements.txt

$ python main.py
```

## Results

**Original Noisy Image**

![Original Image](/demo_data/test.png)

**Denoised Image**

![Denoised Image](/demo_data/test_output.png)


[1]: https://github.com/ankitkataria/Image-Denoising-Pipeline/blob/dip-project/Slides_Project.pdf
[2]: https://github.com/ankitkataria/Image-Denoising-Pipeline/blob/dip-project/Report_Project.pdf
