// Reference-image preparation, isomorphic to the reference `pipeline.py`:
//   - every reference: EXIF transpose, `convert("RGB")` (alpha dropped, not composited);
//   - a SINGLE reference: fitted to ≤ 4 MiP (sides 64–4096, Python `round`), LANCZOS-resized, then reflect-padded
//     (centred) to its canvas, the fitted size rounded up to 16. The output is cropped back to the fitted box when the
//     caller gives no size;
//   - each reference of a MULTI-reference edit: LANCZOS under 1 MiP (sides floored), centre-cropped to the 16-px grid;
//   - pixels to [-1, 1] as x / 127.5 − 1, back as uint8(truncate((clamp(x) + 1) · 127.5)).

import CoreGraphics
import Foundation
import ImageIO
import MLX

/// RGB8, interleaved.
public struct Flux3RGBImage: Sendable {
    public var rgb: [UInt8]
    public var width: Int
    public var height: Int

    public init(rgb: [UInt8], width: Int, height: Int) {
        precondition(rgb.count == width * height * 3)
        self.rgb = rgb
        self.width = width
        self.height = height
    }

    /// `convert("RGB")` of straight RGBA: alpha dropped.
    public init(_ rgba: Flux3RGBAImage) {
        var out = [UInt8](repeating: 0, count: rgba.width * rgba.height * 3)
        for i in 0..<(rgba.width * rgba.height) {
            out[i * 3] = rgba.rgba[i * 4]; out[i * 3 + 1] = rgba.rgba[i * 4 + 1]; out[i * 3 + 2] = rgba.rgba[i * 4 + 2]
        }
        self.init(rgb: out, width: rgba.width, height: rgba.height)
    }

    /// Decode a file, apply its EXIF orientation (`ImageOps.exif_transpose`) and drop alpha.
    public static func read(url: URL) throws -> Flux3RGBImage {
        let image = Flux3RGBImage(try Flux3PNG.read(url: url))
        var orientation = 1
        if let src = CGImageSourceCreateWithURL(url as CFURL, nil),
           let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
           let o = props[kCGImagePropertyOrientation] as? Int {
            orientation = o
        }
        return image.oriented(orientation)
    }

    /// Decode in-memory PNG/JPEG data, apply its EXIF orientation and drop alpha (an engine request's image).
    public static func read(data: Data) throws -> Flux3RGBImage {
        Flux3RGBImage(try Flux3RGBAImage.read(data: data))
    }

    /// PIL `exif_transpose` for EXIF orientation 1–8.
    public func oriented(_ orientation: Int) -> Flux3RGBImage {
        let (px, ow, oh) = Flux3Orientation.apply(rgb, width: width, height: height, channels: 3, orientation: orientation)
        return Flux3RGBImage(rgb: px, width: ow, height: oh)
    }

    /// EXIF orientation of encoded image data (1 when absent).
    static func orientation(_ data: Data) -> Int {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let o = props[kCGImagePropertyOrientation] as? Int else { return 1 }
        return o
    }
}

/// `ImageOps.exif_transpose` over interleaved pixels of any channel count.
enum Flux3Orientation {
    static func apply(_ px: [UInt8], width w: Int, height h: Int, channels n: Int, orientation: Int) -> ([UInt8], Int, Int) {
        guard (2...8).contains(orientation) else { return (px, w, h) }
        let swap = orientation >= 5
        let (ow, oh) = swap ? (h, w) : (w, h)
        var out = [UInt8](repeating: 0, count: ow * oh * n)
        for y in 0..<oh {
            for x in 0..<ow {
                let (sx, sy): (Int, Int)
                switch orientation {
                case 2: (sx, sy) = (w - 1 - x, y)            // FLIP_LEFT_RIGHT
                case 3: (sx, sy) = (w - 1 - x, h - 1 - y)    // ROTATE_180
                case 4: (sx, sy) = (x, h - 1 - y)            // FLIP_TOP_BOTTOM
                case 5: (sx, sy) = (y, x)                    // TRANSPOSE
                case 6: (sx, sy) = (y, h - 1 - x)            // ROTATE_270
                case 7: (sx, sy) = (w - 1 - y, h - 1 - x)    // TRANSVERSE
                default: (sx, sy) = (w - 1 - y, x)           // 8: ROTATE_90
                }
                let s = (sy * w + sx) * n, d = (y * ow + x) * n
                for c in 0..<n { out[d + c] = px[s + c] }
            }
        }
        return (out, ow, oh)
    }
}

extension Flux3RGBImage {

    /// PIL `resize((w, h), LANCZOS)`.
    public func resized(width w: Int, height h: Int, filter: Flux3PILResize.Filter = .lanczos) -> Flux3RGBImage {
        if w == width && h == height { return self }
        let out = Flux3PILResize.resampleBands(rgb, bands: 3, width: width, height: height, outWidth: w, outHeight: h,
                                               filter: filter)
        return Flux3RGBImage(rgb: out, width: w, height: h)
    }

    public func cropped(left: Int, top: Int, width w: Int, height h: Int) -> Flux3RGBImage {
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for y in 0..<h {
            let s = ((top + y) * width + left) * 3
            out.replaceSubrange((y * w * 3)..<((y + 1) * w * 3), with: rgb[s..<(s + w * 3)])
        }
        return Flux3RGBImage(rgb: out, width: w, height: h)
    }

    /// numpy `np.pad(..., mode="reflect" | "symmetric")` to (w, h) with the image at (left, top).
    public func padded(width w: Int, height h: Int, left: Int, top: Int, symmetric: Bool) -> Flux3RGBImage {
        func source(_ i: Int, _ n: Int) -> Int {
            if n == 1 { return 0 }
            if symmetric {
                let period = 2 * n
                var m = i % period; if m < 0 { m += period }
                return m < n ? m : period - 1 - m
            }
            let period = 2 * (n - 1)
            var m = i % period; if m < 0 { m += period }
            return m < n ? m : period - m
        }
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for y in 0..<h {
            let sy = source(y - top, height)
            for x in 0..<w {
                let s = (sy * width + source(x - left, width)) * 3, d = (y * w + x) * 3
                out[d] = rgb[s]; out[d + 1] = rgb[s + 1]; out[d + 2] = rgb[s + 2]
            }
        }
        return Flux3RGBImage(rgb: out, width: w, height: h)
    }

    /// [1, 3, H, W] float32 in [-1, 1]: x / 127.5 − 1 (reference `to_tensor`).
    public func pixelsNCHW() -> MLXArray {
        let plane = width * height
        var out = [Float](repeating: 0, count: 3 * plane)
        for i in 0..<plane { for c in 0..<3 { out[c * plane + i] = Float(rgb[i * 3 + c]) / 127.5 - 1 } }
        return MLXArray(out, [1, 3, height, width])
    }

    /// From [3, H, W] in [-1, 1]: uint8 of (clamp + 1) · 127.5, truncated (reference `to_pil`).
    public static func fromPixels(_ chw: MLXArray) -> Flux3RGBImage {
        let (h, w) = (chw.dim(1), chw.dim(2))
        let v = ((clip(chw.asType(.float32), min: -1, max: 1) + 1) * 127.5).transposed(1, 2, 0).asType(.uint8)
        return Flux3RGBImage(rgb: v.asArray(UInt8.self), width: w, height: h)
    }

    public func rgba() -> Flux3RGBAImage {
        var out = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            out[i * 4] = rgb[i * 3]; out[i * 4 + 1] = rgb[i * 3 + 1]; out[i * 4 + 2] = rgb[i * 3 + 2]
        }
        return Flux3RGBAImage(rgba: out, width: width, height: height)
    }

    public func write(to url: URL) throws { try Flux3PNG.write(rgba(), to: url) }
}

extension Flux3RGBAImage {
    /// Decode in-memory PNG/JPEG data with its EXIF orientation applied, alpha kept (the upsampler composites it over
    /// white).
    public static func read(data: Data) throws -> Flux3RGBAImage {
        let image = try Flux3PNG.read(data: data)
        let (px, w, h) = Flux3Orientation.apply(image.rgba, width: image.width, height: image.height, channels: 4,
                                                orientation: Flux3RGBImage.orientation(data))
        return Flux3RGBAImage(rgba: px, width: w, height: h)
    }
}

public enum Flux3References {
    public static let downsamplingRatio = 16
    public static let matchMaxPixels = 4 * 1024 * 1024
    public static let minSide = 64, maxSide = 4096
    public static let maxReferences = 10
    public static let multiReferenceMaxPixels = 1024 * 1024

    static func ceil16(_ v: Int) -> Int { (v + downsamplingRatio - 1) / downsamplingRatio * downsamplingRatio }

    /// `reference_canvas`: (fitted size, canvas) for a single reference. `maxPixels` is the reference's 4 MiP unless a
    /// caller (an engine package) caps it to the envelope its footprint was measured at.
    public static func canvas(width: Int, height: Int, maxPixels: Int = matchMaxPixels) -> (fit: (Int, Int), canvas: (Int, Int)) {
        var scale = min(1.0, (Double(maxPixels) / Double(width * height)).squareRoot(),
                        Double(maxSide) / Double(max(width, height)))
        while true {
            let fit = (max(1, Int((Double(width) * scale).rounded(.toNearestOrEven))),
                       max(1, Int((Double(height) * scale).rounded(.toNearestOrEven))))
            let canvas = (min(maxSide, max(minSide, ceil16(fit.0))), min(maxSide, max(minSide, ceil16(fit.1))))
            if canvas.0 * canvas.1 <= maxPixels { return (fit, canvas) }
            scale *= 0.995
        }
    }

    /// The crop box (left, top, width, height) of the fitted image inside its padded canvas.
    public struct Box: Sendable { public var left, top, width, height: Int }

    /// `prepare_reference`: one reference, fitted and reflect-padded; the box is nil when no padding was needed.
    public static func prepareSingle(_ image: Flux3RGBImage, maxPixels: Int = matchMaxPixels)
        -> (image: Flux3RGBImage, box: Box?) {
        let (fit, canvas) = canvas(width: image.width, height: image.height, maxPixels: maxPixels)
        var img = image
        if fit != (image.width, image.height) { img = img.resized(width: fit.0, height: fit.1) }
        if canvas == (img.width, img.height) { return (img, nil) }
        let left = (canvas.0 - img.width) / 2, top = (canvas.1 - img.height) / 2
        let symmetric = !(min(img.width, img.height) > downsamplingRatio)
        let padded = img.padded(width: canvas.0, height: canvas.1, left: left, top: top, symmetric: symmetric)
        return (padded, Box(left: left, top: top, width: img.width, height: img.height))
    }

    /// `prepare_multi_reference`: under 1 MiP (sides floored), centre-cropped to the 16-px grid.
    public static func prepareMulti(_ image: Flux3RGBImage) -> Flux3RGBImage {
        var img = image
        var (w, h) = (image.width, image.height)
        if w * h > multiReferenceMaxPixels {
            let s = (Double(multiReferenceMaxPixels) / Double(w * h)).squareRoot()
            (w, h) = (Int(Double(w) * s), Int(Double(h) * s))
            img = img.resized(width: w, height: h)
        }
        let (cw, ch) = (w / downsamplingRatio * downsamplingRatio, h / downsamplingRatio * downsamplingRatio)
        return img.cropped(left: (w - cw) / 2, top: (h - ch) / 2, width: cw, height: ch)
    }
}
