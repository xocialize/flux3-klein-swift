// Bounding boxes for FLUX 3 Klein's layout mode — a port of the reference `bbox.py` (bfc8889).
//
// Boxes are [y_min, x_min, y_max, x_max]: 0–1 from a caller, 0–1000 on the "canvas" the upsampler and the DiT read.
// A box table follows the caption as a JSON list; each row is {"id", "bbox", "desc"} (layout) or {"id", "from",
// "src_bbox", "tgt_bbox", "desc"} (box edits; a null target removes the element). Rows serialize exactly as
// Python's `json.dumps(rows, ensure_ascii=False)`: ", " / ": " separators, keys in insertion order, non-ASCII kept.

import Foundation

/// A JSON value with insertion-ordered objects (Python dict semantics), so a table round-trips as the reference writes it.
public indirect enum Flux3JSON: Equatable, Sendable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([Flux3JSON])
    case object([(String, Flux3JSON)])

    public static func == (a: Flux3JSON, b: Flux3JSON) -> Bool {
        switch (a, b) {
        case (.null, .null): return true
        case let (.bool(x), .bool(y)): return x == y
        case let (.int(x), .int(y)): return x == y
        case let (.double(x), .double(y)): return x == y
        case let (.string(x), .string(y)): return x == y
        case let (.array(x), .array(y)): return x == y
        case let (.object(x), .object(y)): return x.count == y.count && zip(x, y).allSatisfy { $0.0 == $1.0 && $0.1 == $1.1 }
        default: return false
        }
    }

    subscript(key: String) -> Flux3JSON? {
        if case .object(let pairs) = self { return pairs.first { $0.0 == key }?.1 }
        return nil
    }

    var string: String? { if case .string(let s) = self { return s }; return nil }
    var isNull: Bool { if case .null = self { return true }; return false }
    var number: Double? {
        switch self {
        case .int(let i): return Double(i)
        case .double(let d): return d
        default: return nil
        }
    }

    /// `json.dumps(value, ensure_ascii=False)`.
    public func dumps() -> String {
        switch self {
        case .null: return "null"
        case .bool(let b): return b ? "true" : "false"
        case .int(let i): return String(i)
        case .double(let d):
            if d == d.rounded() && abs(d) < 1e16 { return String(format: "%.1f", d) }
            return String(d)
        case .string(let s): return Self.quote(s)
        case .array(let items): return "[" + items.map { $0.dumps() }.joined(separator: ", ") + "]"
        case .object(let pairs): return "{" + pairs.map { Self.quote($0.0) + ": " + $0.1.dumps() }.joined(separator: ", ") + "}"
        }
    }

    static func quote(_ s: String) -> String {
        var out = "\""
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if scalar.value < 0x20 { out += String(format: "\\u%04x", scalar.value) } else { out.unicodeScalars.append(scalar) }
            }
        }
        return out + "\""
    }

    /// Parses one JSON value spanning ALL of `text` (surrounding whitespace allowed), or nil.
    public static func parse(_ text: String) -> Flux3JSON? {
        var p = Parser(Array(text.unicodeScalars))
        p.skipWS()
        guard let v = p.value() else { return nil }
        p.skipWS()
        return p.i == p.s.count ? v : nil
    }

    struct Parser {
        let s: [Unicode.Scalar]
        var i = 0
        init(_ s: [Unicode.Scalar]) { self.s = s }

        mutating func skipWS() { while i < s.count, [" ", "\n", "\r", "\t"].contains(s[i]) { i += 1 } }

        mutating func value() -> Flux3JSON? {
            guard i < s.count else { return nil }
            switch s[i] {
            case "{":
                i += 1; skipWS()
                var pairs: [(String, Flux3JSON)] = []
                if i < s.count, s[i] == "}" { i += 1; return .object(pairs) }
                while true {
                    skipWS()
                    guard case .string(let k)? = string() else { return nil }
                    skipWS(); guard i < s.count, s[i] == ":" else { return nil }; i += 1; skipWS()
                    guard let v = value() else { return nil }
                    if let at = pairs.firstIndex(where: { $0.0 == k }) { pairs[at].1 = v } else { pairs.append((k, v)) }
                    skipWS()
                    guard i < s.count else { return nil }
                    if s[i] == "," { i += 1; continue }
                    if s[i] == "}" { i += 1; return .object(pairs) }
                    return nil
                }
            case "[":
                i += 1; skipWS()
                var items: [Flux3JSON] = []
                if i < s.count, s[i] == "]" { i += 1; return .array(items) }
                while true {
                    skipWS()
                    guard let v = value() else { return nil }
                    items.append(v); skipWS()
                    guard i < s.count else { return nil }
                    if s[i] == "," { i += 1; continue }
                    if s[i] == "]" { i += 1; return .array(items) }
                    return nil
                }
            case "\"": return string()
            case "t": return literal("true", .bool(true))
            case "f": return literal("false", .bool(false))
            case "n": return literal("null", .null)
            case "N": return literal("NaN", .double(.nan))
            case "I": return literal("Infinity", .double(.infinity))
            default: return number()
            }
        }

        mutating func literal(_ word: String, _ v: Flux3JSON) -> Flux3JSON? {
            let w = Array(word.unicodeScalars)
            guard i + w.count <= s.count, Array(s[i..<(i + w.count)]) == w else { return nil }
            i += w.count
            return v
        }

        mutating func number() -> Flux3JSON? {
            let start = i
            if i < s.count, s[i] == "-" {
                i += 1
                if i + 8 <= s.count, String(String.UnicodeScalarView(s[i..<(i + 8)])) == "Infinity" {
                    i += 8; return .double(-.infinity)
                }
            }
            var isFloat = false
            while i < s.count, ("0"..."9").contains(s[i]) || [".", "e", "E", "+", "-"].contains(s[i]) {
                if [".", "e", "E"].contains(s[i]) { isFloat = true }
                i += 1
            }
            let text = String(String.UnicodeScalarView(s[start..<i]))
            if text.isEmpty || text == "-" { return nil }
            if !isFloat, let n = Int(text) { return .int(n) }
            guard let d = Double(text) else { return nil }
            return .double(d)
        }

        mutating func string() -> Flux3JSON? {
            guard i < s.count, s[i] == "\"" else { return nil }
            i += 1
            var out = String.UnicodeScalarView()
            while i < s.count {
                let c = s[i]; i += 1
                if c == "\"" { return .string(String(out)) }
                if c.value < 0x20 { return nil }  // Python's strict mode rejects raw control characters
                if c != "\\" { out.append(c); continue }
                guard i < s.count else { return nil }
                let e = s[i]; i += 1
                switch e {
                case "\"": out.append("\""); case "\\": out.append("\\"); case "/": out.append("/")
                case "b": out.append("\u{08}"); case "f": out.append("\u{0C}"); case "n": out.append("\n")
                case "r": out.append("\r"); case "t": out.append("\t")
                case "u":
                    guard let hi = hex4() else { return nil }
                    if (0xD800...0xDBFF).contains(hi), i + 1 < s.count, s[i] == "\\", s[i + 1] == "u" {
                        i += 2
                        guard let lo = hex4(), (0xDC00...0xDFFF).contains(lo),
                              let u = Unicode.Scalar(0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)) else { return nil }
                        out.append(u)
                    } else if let u = Unicode.Scalar(hi) { out.append(u) } else { return nil }
                default: return nil
                }
            }
            return nil
        }

        mutating func hex4() -> UInt32? {
            guard i + 4 <= s.count, let v = UInt32(String(String.UnicodeScalarView(s[i..<(i + 4)])), radix: 16) else { return nil }
            i += 4
            return v
        }
    }
}

/// A caller's box: 0–1 coordinates. `bbox` places an element (layout); `from` / `srcBBox` / `tgtBBox` move, replace or
/// (with a nil target) remove one in a box edit.
public struct Flux3Box: Sendable, Equatable {
    public var id: String?
    public var desc: String?
    public var bbox: [Double]?
    public var from: String?
    public var srcBBox: [Double]?
    public var tgtBBox: [Double]?

    public init(id: String? = nil, desc: String? = nil, bbox: [Double]? = nil, from: String? = nil,
                srcBBox: [Double]? = nil, tgtBBox: [Double]? = nil) {
        self.id = id; self.desc = desc; self.bbox = bbox; self.from = from; self.srcBBox = srcBBox; self.tgtBBox = tgtBBox
    }
}

public enum Flux3Boxes {
    public static let scale = 1000.0

    static func isID(_ s: String) -> Bool {
        guard let f = s.unicodeScalars.first, f.isASCII, CharacterSet.letters.contains(f) else { return false }
        return s.unicodeScalars.allSatisfy { $0.isASCII && (CharacterSet.alphanumerics.contains($0) || $0 == "_") }
    }

    /// `to_canvas`: 0–1 boxes → rows on the 0–1000 canvas.
    public static func toCanvas(_ boxes: [Flux3Box]) throws -> [Flux3JSON] {
        func grid(_ box: [Double]?) throws -> Flux3JSON {
            guard let box, !box.isEmpty else { return .null }
            guard box.count == 4 else { throw Flux3KleinError.invalidInput("a box is [y_min, x_min, y_max, x_max], got \(box)") }
            return .array(box.map { .int(Int((max(0, min(1, $0)) * scale).rounded(.toNearestOrEven))) })
        }
        return try boxes.enumerated().map { i, box in
            let id = (box.id?.isEmpty == false) ? box.id! : "obj_\(i + 1)"
            let desc = box.desc ?? ""
            if let b = box.bbox, !b.isEmpty {
                return .object([("id", .string(id)), ("bbox", try grid(b)), ("desc", .string(desc))])
            }
            if (box.srcBBox?.isEmpty == false) || (box.tgtBBox?.isEmpty == false) {
                return .object([("id", .string(id)), ("from", .string(box.from ?? "")), ("src_bbox", try grid(box.srcBBox)),
                                ("tgt_bbox", try grid(box.tgtBBox)), ("desc", .string(desc))])
            }
            throw Flux3KleinError.invalidInput("box \(i) has neither bbox nor src_bbox/tgt_bbox")
        }
    }

    /// `split`: (caption, rows) at the LAST "[" from which the rest parses as a JSON list.
    public static func split(_ text: String) -> (String, [Flux3JSON]?) {
        let scalars = Array(text.unicodeScalars)
        for i in stride(from: scalars.count - 1, through: 0, by: -1) where scalars[i] == "[" {
            let tail = String(String.UnicodeScalarView(scalars[i...]))
            if case .array(let rows)? = Flux3JSON.parse(tail) {
                let head = String(String.UnicodeScalarView(scalars[..<i]))
                return (head.trimmingTrailingWhitespace(), rows)
            }
        }
        return (text.trimmingCharacters(in: .whitespacesAndNewlines), nil)
    }

    /// `valid_box`: a well-formed 0–1000 box, or nil.
    static func validBox(_ v: Flux3JSON?) -> [Double]? {
        guard case .array(let items)? = v, items.count == 4 else { return nil }
        var box: [Double] = []
        for item in items {
            guard let n = item.number, n == n, abs(n) != .infinity else { return nil }
            box.append(n)
        }
        let (y1, x1, y2, x2) = (box[0], box[1], box[2], box[3])
        if y1 < 0 || x1 < 0 || y2 <= y1 || x2 <= x1 || y2 > scale || x2 > scale || box.max()! <= 1 { return nil }
        return box
    }

    /// `detect`: (caption, rows) when the prompt already carries a box table on the 0–1000 canvas.
    public static func detect(_ prompt: String) -> (String, [Flux3JSON])? {
        let (text, rows) = split(prompt)
        guard let rows, !rows.isEmpty, rows.allSatisfy({ row in
            guard case .object = row, let id = row["id"]?.string, isID(id) else { return false }
            return ["bbox", "src_bbox", "tgt_bbox"].contains { validBox(row[$0]) != nil }
        }) else { return nil }
        let prefix = "ORIGINAL PROMPT:\n", marker = "\n\nLOCKED BOUNDING-BOX ANNOTATIONS "
        if text.hasPrefix(prefix), let r = text.range(of: marker) {
            return (String(text[text.index(text.startIndex, offsetBy: prefix.count)..<r.lowerBound]), rows)
        }
        return (text, rows)
    }

    /// `is_finished`: an upsampler answer (its box table cited in full as <id>s), not a request.
    public static func isFinished(_ prompt: String) -> Bool {
        guard let (caption, rows) = detect(prompt) else { return false }
        return rows.allSatisfy { caption.contains("<\($0["id"]?.string ?? "")>") }
    }

    /// `plan`: (system prompt key, user text, locked rows).
    public static func plan(_ prompt: String, boxes: [Flux3Box]?, imageCount: Int) throws -> (String, String, [Flux3JSON]?) {
        let task = imageCount > 0 ? "i2i" : "t2i"
        var prompt = prompt
        let locked: [Flux3JSON]
        if let boxes, !boxes.isEmpty {
            locked = try toCanvas(boxes)
        } else if let (text, rows) = detect(prompt) {
            prompt = text
            locked = rows
        } else {
            return (task, prompt, nil)
        }
        return ("\(task)_bbox", upsamplingInput(prompt, rows: locked), locked)
    }

    /// `upsampling_input`.
    public static func upsamplingInput(_ prompt: String, rows: [Flux3JSON]) -> String {
        "ORIGINAL PROMPT:\n\(prompt.trimmingCharacters(in: .whitespacesAndNewlines))\n\n"
            + "LOCKED BOUNDING-BOX ANNOTATIONS ([y_min, x_min, y_max, x_max] on a 0-1000 canvas, with (0, 0) at "
            + "the top-left). This is an authoritative JSON list of dictionaries; preserve every supplied identity, "
            + "provenance value, null, and coordinate exactly:\n\(Flux3JSON.array(rows).dumps())"
    }

    /// `valid_added`: a row the upsampler added, kept only if cited as <id> and well formed.
    static func validAdded(_ row: Flux3JSON, caption: String) -> Flux3JSON? {
        guard case .object = row, let id = row["id"]?.string, isID(id), caption.contains("<\(id)>"),
              let desc = row["desc"]?.string, !desc.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return nil }
        let d = desc.trimmingCharacters(in: .whitespacesAndNewlines)
        if let box = validBox(row["bbox"]) {
            return .object([("id", .string(id)), ("bbox", original(row["bbox"]!, box)), ("desc", .string(d))])
        }
        let origin = row["from"]?.string, src = row["src_bbox"], tgt = row["tgt_bbox"]
        let srcBox = (src != nil && !src!.isNull) ? validBox(src) : nil
        let tgtBox = (tgt != nil && !tgt!.isNull) ? validBox(tgt) : nil
        let srcGiven = src != nil && !src!.isNull, tgtGiven = tgt != nil && !tgt!.isNull
        guard let origin, origin == "target" || origin.range(of: #"^ref_image_\d+$"#, options: .regularExpression) != nil,
              !(srcGiven && srcBox == nil), !(tgtGiven && tgtBox == nil), !(srcBox == nil && tgtBox == nil),
              (origin == "target") != (srcBox != nil)
        else { return nil }
        return .object([("id", .string(id)), ("from", .string(origin)),
                        ("src_bbox", srcBox == nil ? .null : original(src!, srcBox!)),
                        ("tgt_bbox", tgtBox == nil ? .null : original(tgt!, tgtBox!)), ("desc", .string(d))])
    }

    /// `list(box)` keeps the parsed numbers as they were (ints stay ints, floats stay floats).
    static func original(_ v: Flux3JSON, _ box: [Double]) -> Flux3JSON {
        if case .array = v { return v }
        return .array(box.map { .double($0) })
    }

    static func iou(_ a: [Double], _ b: [Double]) -> Double {
        let inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return union > 0 ? inter / union : 0
    }

    /// `reconcile`: the upsampler's answer with the caller's boxes re-imposed.
    public static func reconcile(_ text: String, locked: [Flux3JSON]) -> String {
        var (caption, parsed) = split(text)
        var rows = parsed ?? []
        let lockedIDs = Set(locked.compactMap { $0["id"]?.string })
        let rowIDs = Set(rows.compactMap { $0["id"]?.string })
        var renamed: [(String, String)] = []
        for lock in locked {
            guard let lockID = lock["id"]?.string else { continue }
            let lockBox = [lock["bbox"], lock["tgt_bbox"], lock["src_bbox"]].compactMap { v -> [Double]? in
                guard let v, !v.isNull, case .array(let a) = v, !a.isEmpty else { return nil }
                return a.compactMap(\.number)
            }.first
            if rowIDs.contains(lockID) || lockBox == nil { continue }
            for row in rows {
                guard case .object = row, let rid = row["id"]?.string else { continue }
                if renamed.contains(where: { $0.0 == rid }) || lockedIDs.contains(rid) { continue }
                let first = [row["bbox"], row["tgt_bbox"], row["src_bbox"]].first { v in
                    guard let v, !v.isNull, case .array(let a) = v else { return false }
                    return !a.isEmpty
                } ?? nil
                if let box = validBox(first), iou(box, lockBox!) >= 0.9 {
                    renamed.append((rid, lockID))
                    break
                }
            }
        }
        rows = rows.map { row in
            guard case .object(var pairs) = row, let rid = row["id"]?.string,
                  let new = renamed.first(where: { $0.0 == rid })?.1,
                  let at = pairs.firstIndex(where: { $0.0 == "id" }) else { return row }
            pairs[at].1 = .string(new)
            return .object(pairs)
        }
        for (old, new) in renamed { caption = caption.replacingOccurrences(of: "<\(old)>", with: "<\(new)>") }

        // Python's dict comprehension keeps the LAST row per id.
        var byID: [String: Flux3JSON] = [:]
        for row in rows { if case .object = row, let id = row["id"]?.string { byID[id] = row } }
        var out: [Flux3JSON] = []
        for lock in locked {
            guard case .object(var pairs) = lock else { continue }
            let id = lock["id"]?.string ?? ""
            let upsampled = byID[id]?["desc"]?.string?.trimmingCharacters(in: .whitespacesAndNewlines)
            let desc: String = (upsampled?.isEmpty == false) ? upsampled! : (lock["desc"]?.string ?? "")
            if let at = pairs.firstIndex(where: { $0.0 == "desc" }) { pairs[at].1 = .string(desc) } else { pairs.append(("desc", .string(desc))) }
            out.append(.object(pairs))
        }
        for row in rows {
            if let added = validAdded(row, caption: caption), let id = added["id"]?.string,
               !out.contains(where: { $0["id"]?.string == id }) {
                out.append(added)
            }
        }
        return "\(caption) \(Flux3JSON.array(out).dumps())"
    }
}

extension String {
    /// Python `str.rstrip()`.
    public func trimmingTrailingSpace() -> String { trimmingTrailingWhitespace() }

    func trimmingTrailingWhitespace() -> String {
        var s = Substring(self)
        while let last = s.unicodeScalars.last, CharacterSet.whitespacesAndNewlines.contains(last) { s = s.dropLast() }
        return String(s)
    }
}
