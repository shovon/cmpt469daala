import Canvas, { Image } from 'canvas';
import fs from 'fs';
import path from 'path';

const blockPadding = 4;
const blockSize = 8;

// https://antimatter15.com/2015/05/cooley-tukey-fft-dct-idct-in-under-1k-of-javascript/

/**
 * Performs the fourier transform on the given signal, defined by both the real
 * and imaginary components
 *
 * @param { number[] } re the real component of the signal
 * @param { number[] } im the imaginary component of the signal
 */
function miniFFT(re, im) {
  var N = re.length;
  for (var i = 0; i < N; i++) {
    for(var j = 0, h = i, k = N; k >>= 1; h >>= 1)
      j = (j << 1) | (h & 1);
    if (j > i) {
      re[j] = [re[i], re[i] = re[j]][0]
      im[j] = [im[i], im[i] = im[j]][0]
    }
  }
  for(var hN = 1; hN * 2 <= N; hN *= 2)
    for (var i = 0; i < N; i += hN * 2)
      for (var j = i; j < i + hN; j++) {
        var cos = Math.cos(Math.PI * (j - i) / hN),
            sin = Math.sin(Math.PI * (j - i) / hN)
        var tre =  re[j+hN] * cos + im[j+hN] * sin,
            tim = -re[j+hN] * sin + im[j+hN] * cos;
        re[j + hN] = re[j] - tre; im[j + hN] = im[j] - tim;
        re[j] += tre; im[j] += tim;
      }
}

/**
 * Performs the inverse of the fourier transform on the given signal, defined by
 * both the real and imaginary components.
 *
 * @param { number[] } re the real component of the signal.
 * @param { number[] } im the imaginary component of the signal.
 */
function miniIFFT(re, im){
  miniFFT(im, re);
  for(var i = 0, N = re.length; i < N; i++){
    im[i] /= N;
    re[i] /= N;
  }
}

/**
 * Performs the discrete cosine transform Type-II on the given signal.
 *
 * @param { number[] } s the signal to perform the transformation on.
 */
function miniDCT(s){
  var N = s.length,
      K = -Math.PI / (2 * N),
      re = new Float64Array(N),
      im = new Float64Array(N);
  for(var i = 0, j = N; j > i; i++){
    re[i] = s[i * 2]
    re[--j] = s[i * 2 + 1]
  }
  miniFFT(re, im)
  for(var i = 0; i < N; i++)
    s[i] = 2*re[i]*Math.cos(K*i)-2*im[i]*Math.sin(K*i);
}

/**
 * Performs the discrete cosine transform Type-II on the given signal.
 *
 * @param { number[] } s the signal to perform the transformation on.
 */
function miniIDCT(s){
  var N = s.length,
      K = Math.PI / (2 * N),
      im = new Float64Array(N),
      re = new Float64Array(N);
  re[0] = s[0] / N / 2;
  for(var i = 1; i < N; i++){
    var im2 = Math.sin(i*K), re2 = Math.cos(i*K);
    re[i] = (s[N - i] * im2 + s[i] * re2) / N / 2;
    im[i] = (im2 * s[i] - s[N - i] * re2) / N / 2;
  }
  miniFFT(im, re)
  for(var i = 0; i < N / 2; i++){
    s[2 * i] = re[i]
    s[2 * i + 1] = re[N - i - 1]
  }
}

/**
 * Clamps a number between the specified max and min.
 *
 * @param { number } val the value
 * @param { max } the maximum value
 * @param { min } the minimum value
 */
function clamp(val, max, min = 0) {
  if (val > max) { return max; }
  if (val < min) { return min; }
  return val;
}

/**
 * Applies the specified transform to the signal.
 *
 * @param { number[] } our signal to transform
 * @param { function } the transform to apply
 * @param { bSize } the block size
 */
function transform2D(s, transform, bSize = blockSize) {
  const holder = Array(blockSize);
  for (let y = 0; y < bSize; y++) {
    for (let x = 0; x < bSize; x++) {
      holder[x] = s[x + y * bSize];
    }
    transform(holder);
    for (let x = 0; x < bSize; x++) {
      s[x + y * bSize] = holder[x];
    }
  }
  for (let x = 0; x < bSize; x++) {
    for (let y = 0; y < bSize; y++) {
      holder[y] = s[x + y * bSize];
    }
    transform(holder);
    for (let y = 0; y < bSize; y++) {
      s[x + y * bSize] = holder[y];
    }
  }
}

const quantizationMatrix = [
 16,  11,  10,  16,  24,  40,  51,  61,
 12,  12,  14,  19,  26,  58,  60,  55,
 14,  13,  16,  24,  40,  57,  69,  56,
 14,  17,  22,  29,  51,  87,  80,  62,
 18,  22,  37,  56,  68,  109, 103, 77,
 24,  35,  55,  64,  81,  104, 113, 92,
 49,  64,  78,  87,  103, 121, 120, 101,
 72,  92,  95,  98,  112, 100, 103, 99
];

// const quality = 100;

/**
 * Gets the quality coefficient to apply.
 *
 * @param { number } q the quality to apply
 * @param { number } index the index in which the coefficient is being applied
 *
 * @returns { number }
 */
function getQualityCoefficient(q, index) {
  if (q === 100) {
    return 150;
  }
  return quantizationMatrix[index] * (1000 / q);
}

/**
 * Applies the DCT Type-II to the specified image's regions.
 *
 * @param { Uint32Array } imageBuffer the image to apply the DCT Type-II to.
 * @param { number } imageWidth the width of the image.
 * @param { startX } startX where the block starts in the X coordinate.
 * @param { startY } startY where the block starts in the Y coordinate.
 * @param { bSize } bSize the block size.
 */
function putDCT(imageBuffer, imageWidth, startX, startY, quality = 100, bSize = blockSize) {
  const arr = Array(blockSize * blockSize);

  for (let x = startX; x < startX + blockSize; x++) {
    for (let y = startY; y < startY + blockSize; y++) {
      const index = x - startX + (y - startY) * bSize;
      arr[index] = (imageBuffer[x + y * imageWidth] & 255) - 128;
    }
  }

  transform2D(arr, miniDCT);

  for (let x = startX; x < startX + blockSize; x++) {
    for (let y = startY; y < startY + blockSize; y++) {
      const index = (x - startX) + (y - startY) * bSize;
      const quantization = getQualityCoefficient(quality, index);
      const b = clamp(Math.round(arr[index] / quantization) + 128, 255);
      const originalAlpha = (imageBuffer[x + y * imageWidth] >> 24) & 255;
      const pixel = b | (b << 8) | (b << 16) | (originalAlpha << 24);
      imageBuffer[x + y * imageWidth] = pixel;
    }
  }
}

/**
 * Applies the DCT Type-III to the specified image's regions.
 *
 * @param { Uint32Array } imageBuffer the image to apply the DCT Type-III to.
 * @param { number } imageWidth the width of the image.
 * @param { startX } startX where the block starts in the X coordinate.
 * @param { startY } startY where the block starts in the Y coordinate.
 * @param { bSize } bSize the block size.
 */
function putIDCT(imageBuffer, imageWidth, startX, startY, quality = 100, bSize = blockSize) {
  const arr = Array(blockSize * blockSize);

  for (let x = startX; x < startX + blockSize; x++) {
    for (let y = startY; y < startY + blockSize; y++) {
      const index = x - startX + (y - startY) * bSize;
      const quantization = getQualityCoefficient(quality, index);
      arr[index] = ((imageBuffer[x + y * imageWidth] & 255) - 128) * quantization;
    }
  }

  transform2D(arr, miniIDCT);

  for (let x = startX; x < startX + blockSize; x++) {
    for (let y = startY; y < startY + blockSize; y++) {
      const index = (x - startX) + (y - startY) * bSize;
      const b = clamp(arr[index] + 128, 255) & 255;
      const originalAlpha = (imageBuffer[x + y * imageWidth] >> 24) & 255;
      const pixel = b | (b << 8) | (b << 16) | (originalAlpha << 24);
      imageBuffer[x + y * imageWidth] = pixel;
    }
  }
}

/**
 * Gets a canvas object such that each block of the new canvas is a DCT of the
 * corresponding block from the original canvas.
 *
 * @param { HTML5CanvasElement } canvas the image that we want the DCT of
 *
 * @return { HTML5CanvasElement }
 */
function getDCTCanvas(canvas, quality = 100) {
  return getBlockTransformCanvas(canvas, putDCT, quality);
}

/**
 * Gets a canvas object such that each block of the new canvas is a DCT Type-III
 * of the corresponding block from the original canvas.
 *
 * @param { HTML5CanvasElement } canvas the image to apply the DCT Type-III
 *
 * @return { HTML5CanvasElement }
 */
function getIDCTCanvas(canvas, quality = 100) {
  return getBlockTransformCanvas(canvas, putIDCT, quality);
}

/**
 * Performs a block transform on the input canvas.
 *
 * @param { HTML5CanvasElement } canvas the input canvas.
 * @param { function } transform the transform that we want to perform
 *
 * @param { HTML5CanvasElement }
 */
function getBlockTransformCanvas(canvas, transform, quality) {
  const context = canvas.getContext('2d');

  const dctCanvas = new Canvas(canvas.width, canvas.height);
  const dctContext = dctCanvas.getContext('2d');
  dctContext.drawImage(canvas, 0, 0);
  const imageData = context.getImageData(
    0,
    0,
    dctCanvas.width,
    dctCanvas.height
  );
  const imageBuffer = new Uint32Array(imageData.data.buffer);

  for (let y = 0; y < canvas.height / blockSize; y++) {
    for (let x = 0; x < canvas.width / blockSize; x++) {
      transform(imageBuffer, canvas.width, x * blockSize, y * blockSize, quality);
    }
  }

  dctContext.putImageData(imageData, 0, 0);

  return dctCanvas;
}

/**
 * Returns a greyscale equivalent of the supplied pixel. Does not modify the
 * alpha value.
 *
 * @param { number } pixel the pixel to get the greyscale of.
 *
 * @return { number }
 */
function convertPixelToGreyscale(pixel) {
  const r = pixel & 255;
  const g = (pixel >> 8) & 255;
  const b = (pixel >> 16) & 255;
  const a = (pixel >> 24) & 255;
  const average = Math.floor((r + g + b) / 3);
  const newR = average;
  const newG = average;
  const newB = average;
  const newPixel = newR | (newG << 8) | (newB << 16) | (a << 24);
  return newPixel;
}

/**
 * Takes the pixels of the input canvas, and returns a greyscale equivalent.
 *
 * @param { HTML5CanvasElement } the canvas that has our image.
 *
 * @return { HTML5CanvasElement } the canvas that has our greyscale analog
 */
function getGreyscaleCanvas(canvas) {
  const context = canvas.getContext('2d');
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
  const imageBuffer = new Uint32Array(imageData.data.buffer);

  const greyscaleCanvas = new Canvas(canvas.width, canvas.height);
  const greyscaleContext = greyscaleCanvas.getContext('2d');
  const greyscaleContextData = greyscaleContext
    .getImageData(0, 0, canvas.width, canvas.height);
  const greyscaleContextBuffer = new Uint32Array(
    greyscaleContextData.data.buffer
  );

  for (var i = 0; i < imageBuffer.length; i++) {
    greyscaleContextBuffer[i] = convertPixelToGreyscale(imageBuffer[i]);
  }

  greyscaleContext.putImageData(greyscaleContextData, 0, 0);
  return greyscaleCanvas;
}

/**
 * Gets an image such that it has been visually split into blocks. The resulting
 * image is larger, dimension-wise, to compensate for the slits between blocks.
 *
 * @param { HTML5CanvasElement } imageCanvas the canvas that has our image
 * @param { number } blockSize the size of the block
 * @param { number } blockPadding the padding between blocks
 *
 * @return { HTML5CanvasElement }
 */
function getSplittedImageCanvas(
  imageCanvas, bSize = blockSize, bPadding = blockPadding
) {
  const imageContext = imageCanvas.getContext('2d');
  const imageData = imageContext.getImageData(0, 0, image.width, image.height);
  const imageArray = new Uint32Array(imageData.data.buffer);

  const splitCanvas = new Canvas(
    imageCanvas.width + ((imageCanvas.width / bSize) - 1) * bPadding,
    imageCanvas.height + ((imageCanvas.height / bSize) - 1) * bPadding
  );
  const splitContext = splitCanvas.getContext('2d');

  const splitImageData = splitContext
    .getImageData(
      0,
      0,
      splitCanvas.width,
      splitCanvas.height
    );
  const splitImageArray = new Uint32Array(splitImageData.data.buffer);

  for (let y = 0; y < image.height; y++) {
    for (let x = 0; x < image.width; x++) {
      const index = x + y * image.width;
      const blockX = Math.floor(x / blockSize);
      const blockY = Math.floor(y / blockSize);
      const outputX = x + blockX * blockPadding;
      const outputY = y + blockY * blockPadding;
      const outputIndex = outputX + outputY * splitCanvas.width;
      splitImageArray[outputIndex] = imageArray[index];
    }
  }

  splitContext.putImageData(splitImageData, 0, 0);

  return splitCanvas;
}

/**
 * Outputs the canvas' content to the specified file as a PNG.
 *
 * @param { HTML5CanvasElement } canvas the canvas to get the image from.
 * @param { string } filename the filename where the image is being written
 *
 * @return { Promise }
 */
function outputCanvasToFile(canvas, filename) {
  console.log(filename);
  return new Promise((resolve, reject) => {
    canvas.toBuffer((err, buf) => {
      if (err) {
        reject(err);
        return;
      }
      fs.writeFileSync(filename, buf);
      resolve();
    });
  });
}

/**
 * Gets a block from the specified canvas.
 *
 * @param { HTML5CanvasElement } canvas the canvas that we want to get the block
 * @param { number } blockX the x-coordinate of the block
 * @param { number } blockY the y-coordinate of the block
 * @param { number } scale the size of the block in the output
 *
 * @return { HTML5CanvasElement }
 */
function getBlockFromCanvas(canvas, blockX, blockY, scale, bSize = blockSize) {
  const x = blockX * bSize;
  const y = blockY * bSize;

  const context = canvas.getContext('2d');
  const data = context.getImageData(x, y, bSize, bSize);

  const blockCanvas = new Canvas(bSize, bSize);
  const blockContext = blockCanvas.getContext('2d');

  blockContext.putImageData(data, 0, 0, 0, 0, bSize, bSize);

  const outSize = bSize * scale;
  const outCanvas = new Canvas(outSize, outSize);
  const outContext = outCanvas.getContext('2d');
  outContext.imageSmoothingEnabled = false;
  outContext.mozImageSmoothingEnabled = false;
  outContext.webkitImageSmoothingEnabled = false;
  outContext.msImageSmoothingEnabled = false;
  outContext.drawImage(blockCanvas, 0, 0, outSize, outSize);

  return outCanvas;
}

/**
 * Gets a random integer
 */
function getRandomInt(max = 10, min = 0) {
  return Math.floor(Math.random() * (max - min) - min);
}

const commandArgs = process.argv.slice(2);

if (commandArgs.length < 2) {
  console.error('Expecting a file name and an output directory');
  process.exit(1);
}

const [ inputFilename, outputDir ] = commandArgs;
const greyscaleFilename = path.resolve(outputDir, 'grayscale-image.png');
const splitImageFilename = path.resolve(outputDir, 'split-image.png');
const dctImageFilename = path.resolve(outputDir, 'dct-image.png');
const dctImageSplitFilename = path.resolve(outputDir, 'dct-split-image.png');
const dctQuantizedImageFilename = path.resolve(outputDir, 'dct-quantized-image.png');
const dctQuantizedImageSplitFilename = path.resolve(outputDir, 'dct-quantized-split-image.png');
const idctQuantizedImageFilename = path.resolve(outputDir, 'idct-quantized-image.png');
const idctQuantizedImageSplitFilename = path.resolve(outputDir, 'idct-quantized-split-image.png');
const idctImageFilename = path.resolve(outputDir, 'idct-image.png');
const blockImageFilename = path.resolve(outputDir, 'block-image.png');
const blockDCTImageFilename = path.resolve(outputDir, 'block-dct-image.png');
const blockDCTQuantizedFilename = path.resolve(outputDir, 'block-dct-quantized-image.png');

const inputFile = (() => {
  try {
    const file = fs.readFileSync(inputFilename);
    return file;
  } catch (e) {
    console.error('Error reading file');
    process.exit(1);
  }
})();

const image = new Image();
image.src = inputFile;

const imageCanvas = new Canvas(
  image.width,
  image.height
);

const imageContext = imageCanvas.getContext('2d');
imageContext.drawImage(image, 0, 0, image.width, image.height);

const blockX = getRandomInt(512 / 8);
const blockY = getRandomInt(512 / 8);
const blockScale = 64;

const greyscaleCanvas = getGreyscaleCanvas(imageCanvas);
const splitCanvas = getSplittedImageCanvas(greyscaleCanvas);
const dctCanvas = getDCTCanvas(greyscaleCanvas);
const dctSplitCanvas = getSplittedImageCanvas(dctCanvas);
const idctCanvas = getIDCTCanvas(dctCanvas);
const dctQuantizedCanvas = getDCTCanvas(greyscaleCanvas, 10);
const dctQuantizedSplitCanvas = getSplittedImageCanvas(dctQuantizedCanvas);
const idctQuantizedCanvas = getIDCTCanvas(dctQuantizedCanvas, 10);
const idctQuantizedSplittedCanvas = getSplittedImageCanvas(idctQuantizedCanvas);
const blockCanvas = getBlockFromCanvas(greyscaleCanvas, blockX, blockY, blockScale);
const dctBlockCanvas = getBlockFromCanvas(dctCanvas, blockX, blockY, blockScale);
const dctQuantizedBlockCanvas = getBlockFromCanvas(dctQuantizedCanvas, blockX, blockY, blockScale);

Promise.resolve()
  .then(() => outputCanvasToFile(greyscaleCanvas, greyscaleFilename))
  .then(() => outputCanvasToFile(splitCanvas, splitImageFilename))
  .then(() => outputCanvasToFile(dctCanvas, dctImageFilename))
  .then(() => outputCanvasToFile(dctSplitCanvas, dctImageSplitFilename))
  .then(() => outputCanvasToFile(idctCanvas, idctImageFilename))
  .then(() => outputCanvasToFile(blockCanvas, blockImageFilename))
  .then(() => outputCanvasToFile(dctQuantizedCanvas, dctQuantizedImageFilename))
  .then(() => outputCanvasToFile(dctQuantizedSplitCanvas, dctQuantizedImageSplitFilename))
  .then(() => outputCanvasToFile(idctQuantizedCanvas, idctQuantizedImageFilename))
  .then(() => outputCanvasToFile(idctQuantizedSplittedCanvas, idctQuantizedImageSplitFilename))
  .then(() => outputCanvasToFile(dctBlockCanvas, blockDCTImageFilename))
  .then(() => outputCanvasToFile(dctQuantizedBlockCanvas, blockDCTQuantizedFilename));
