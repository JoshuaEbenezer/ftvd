# % out = ADM2TVL2(H,img,mu,opts)
# %
# % Alternating Directions Method (ADM) applied to TV/L2.
# %
# % Suppose the data accuquisition model is given by: img = K*Xbar + Noise,
# % where Xbar is an original image, K is a convolution matrix, Noise is
# % additive noise, and img is a blurry and noisy observation. To recover
# % Xbar from img and K, we solve TV/L2 (ROF) model
# %
# % ***     min_X \sum_i ||Di*X|| + mu/2*||K*X - img||^2      ***
# %
# % Inputs:
# %         H  ---  convolution kernel representing K
# %         img  ---  blurry and noisy observation
# %         mu ---  model prameter (must be provided by user)
# %         opts --- a structure containing algorithm parameters {default}
# %                 * opst.beta_iter    : a positive constant {10}
# %                 * opst.gamma   : a constant in (0,1.618] {1.618}
# %                 * opst.maxitr  : maximum iteration number {500}
# %                 * opst.relchg  : a small positive parameter which controls
# %                                  stopping rule of the code. When the
# %                                  relative change of X is less than
# %                                  opts.relchg, then the code stops. {1.e-3}
# %                 * opts.print   : print inter results or not {1}
# %
# % Outputs:
# %         out --- a structure contains the following fields
# %                * out.snr   : SNR values at each iteration
# %                * out.img     : function valuse at each itertion
# %                * out.relchg: the history of relative change in X
# %                * out.sol   : numerical solution obtained by this code
# %                * out.itr   : number of iterations used
# %





import numpy as np
import cv2
import os
import argparse
import glob
# import adm2tvl1


def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)
    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img

def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf_pad = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf_pad = np.roll(psf_pad, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf_pad)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf_pad.size * np.log2(psf_pad.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf
    
def Dive(X,Y):
    # % Transpose of the forward finite difference operator
    fwd_diff_rowX = np.expand_dims(X[:,-1] - X[:, 1],axis=1)
    DtXY = np.concatenate((fwd_diff_rowX, -np.diff(X,1,1)),axis=1)
    fwd_diff_rowY = np.expand_dims(Y[-1,:] - Y[1, :],axis=0)
    DtXY = DtXY + np.concatenate((fwd_diff_rowY, -np.diff(Y,1,0)),axis=0)
    return DtXY


def getC(image,kernel):
    sizeF = np.shape(image)
    eigsK = psf2otf(kernel,sizeF)   # discrete fourier transform of kernel (equal to eigenvalues due to block circular prop.)
    KtF = np.real(np.fft.ifft2(np.conj(eigsK) * np.fft.fft2(image)))    # equation 2.4 and 2.3 in Wang et al. -> transpose(kernel) * image 
    diff_kernelX = np.expand_dims(np.array([1,-1]),axis=1)
    diff_kernelY = np.expand_dims(np.array([[1],[-1]]),axis=0)
    eigsDtD = np.abs(psf2otf(diff_kernel,sizeF))**2 + np.abs(psf2otf(diff_kernel,sizeF))**2 # Fourier transform of D transpose * D
    eigsKtK = np.abs(eigsK)**2 # Fourier transform of K transpose * K
    return (eigsK,KtF,eigsDtD,eigsKtK)
def ForwardD(U):
    # % Forward finite difference operator
    end_col_diff = np.expand_dims((U[:,0]- U[:,-1]),axis=1)
    end_row_diff = np.expand_dims((U[0,:] - U[-1,:]),axis=0)
    Dux = np.concatenate((np.diff(U,1,1), end_col_diff),axis=1)     # discrete gradient operators
    Duy = np.concatenate((np.diff(U,1,0), end_row_diff),axis=0)
    return (Dux,Duy)
def fval(D1X,D2X,eigsK,X,img,mu):
    f = np.sum(np.sum(np.sqrt(D1X**2 + D2X**2)))   
    KXF = np.real(np.fft.ifft2(eigsK * np.fft.fft2(X))) - img
    f = f + mu/2 * np.linalg.norm(KXF,'fro')**2
    return f

def ftvd(image_folder,beta=10,gamma=1.618,max_itr=500,relchg=1e-3):
    beta_iter = beta
    kernel = np.expand_dims(np.array([1]),axis=1)
    mu = 500
    for fn in glob.glob(image_folder+'/*.jpg'):
        print(fn)
        img = cv2.imread(fn,0)
        img = cv2.resize(img,(128,128))
        Lam1 = np.zeros((img.shape))
        Lam2 = Lam1
        eigsK,KtF,eigsDtD,eigsKtK = getC(img,kernel)

        X = img
        D1X,D2X = ForwardD(X)
        # f = fval(D1X,D2X,eigsK,X,img,mu)

        for ii in range(max_itr):
            # Shrinkage -> equation 2.3
            Z1 = D1X + Lam1/beta_iter   # x component of derivative
            Z2 = D2X + Lam2/beta_iter   # y component of derivative
            V = Z1**2 + Z2**2
            V = np.sqrt(V)
            V[V==0]=1
            V = np.max(V-1/beta_iter,0)/V   # equation 2.2 - optimization for w
            Y1 = Z1*V
            Y2 = Z2*V

            # X subproblem -> equation 2.4
            Xp = X
            X = (mu*KtF - Dive(Lam1,Lam2))/beta_iter + Dive(Y1,Y2)  # Dive (Y1,Y2) = D1*w1+D2*w2
            X = np.fft.fft2(X)/(eigsDtD + (mu/beta_iter)*eigsKtK)   # denominator of 2.4
            X = np.real(np.fft.ifft2(X))    # fourier inversers in 2.4
            relchg_iter = np.linalg.norm(Xp-X,'fro')/np.linalg.norm(Xp,'fro')
            print('Iteration ', ii , ' Relative change: ', relchg_iter)

            # check stopping rule
            if relchg_iter < relchg:                
                solution = X
                number_iter = ii
                D1X,D2X = ForwardD(X)
                break
            else:
                D1X,D2X = ForwardD(X)
                # f = fval(D1X,D2X,eigsK,X,img,mu)
    
                #  Update Lam
                Lam1 = Lam1 - gamma*beta_iter*(Y1 - D1X)
                Lam2 = Lam2 - gamma*beta_iter*(Y2 - D2X)
            beta_iter = beta_iter
                

        solution = X;
        number_iter = ii;
        print('Maximum number of iterations reached')
        name = os.path.basename(fn).split('.')[0] + '.jpg'
        filename = './textured/100_'+name
        cv2.imwrite(filename,solution)

if __name__ == "__main__":
    ftvd('../segmentation/out_images/')
