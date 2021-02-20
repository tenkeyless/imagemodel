# Category Segmentation

Segmentation indicating which category each pixel in the image belongs to.

## Model

* Inputs
  * [`width`, `height`, 3] or [`width`, `height`, 1]
    * RGB Image or Grayscale Image
* Outputs
  * [`width`, `height`, `number of category`]
    * A category for each pixel in image.
