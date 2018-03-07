import cv2
from pylab import *
from sklearn.mixture import GaussianMixture
from PIL import Image

def segViaGMM(pil_image, segCol1, segCol2, startX, startY):
    pil_image = pil_image.convert('RGB')
    im = np.array(pil_image)
    im = im[:, :, ::-1].copy()
    width, height = pil_image.size
    newdata = im.reshape(width*height, 3)
    gmm = GaussianMixture(n_components=2, covariance_type="tied")
    gmm = gmm.fit(newdata)
    cluster = gmm.predict(newdata)
    data = np.zeros((height,width, 3), dtype=np.uint8)
    cluster = np.reshape(cluster,(height,width))
    for (x,y,z), value in np.ndenumerate(data):
        if(cluster[x,y] == cluster[startY,startX]):
            data[x, y] = segCol1
        else:
            data[x,y] = segCol2

    img = Image.fromarray(data, 'RGB')
    return(img)
