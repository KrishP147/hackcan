import cld from './cloudinary';
import { fill, thumbnail } from '@cloudinary/url-gen/actions/resize';
import { auto as autoQuality } from '@cloudinary/url-gen/qualifiers/quality';
import { auto as autoFormat } from '@cloudinary/url-gen/qualifiers/format';
import { autoGravity } from '@cloudinary/url-gen/qualifiers/gravity';

/** Build a CloudinaryVideo asset ready for AdvancedVideo */
export function getVideo(publicId: string) {
  return cld.video(publicId);
}

/** Build a CloudinaryImage from a video frame (poster / thumbnail) */
export function getVideoThumbnail(publicId: string, width = 1280, height = 720) {
  return cld
    .image(publicId)
    .resize(fill().width(width).height(height).gravity(autoGravity()))
    .delivery(autoQuality())
    .format(autoFormat());
}

/** Small square preview thumbnail */
export function getPreviewThumbnail(publicId: string, size = 160) {
  return cld
    .image(publicId)
    .resize(thumbnail().width(size).height(size).gravity(autoGravity()))
    .delivery(autoQuality())
    .format(autoFormat());
}
