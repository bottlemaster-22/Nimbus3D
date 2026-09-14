//
//  JSONValue.swift
//  Export
//
//  A tiny, deterministic JSON writer. `Foundation.JSONSerialization` would
//  work too, but it serializes `[String: Any]` via `Dictionary`, whose key
//  order is not guaranteed stable across runs/OS versions - and this
//  module's mandate is deterministic serialization. This type is an ordered
//  key-value list instead, so a glTF document written twice from the same
//  `SplatCloud` is byte-for-byte identical, which also makes the round-trip
//  self-tests in ExportSelfTest.swift meaningful.
//

import Foundation

indirect enum JSONValue {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case array([JSONValue])
    case object([(String, JSONValue)])

    func serialize() -> String {
        var out = ""
        write(into: &out)
        return out
    }

    private func write(into out: inout String) {
        switch self {
        case .string(let s):
            out.append("\"")
            out.append(Self.escape(s))
            out.append("\"")
        case .int(let i):
            out.append(String(i))
        case .double(let d):
            if d.isFinite {
                out.append(Self.formatDouble(d))
            } else {
                out.append("0")  // glTF/JSON has no NaN/Infinity; never emit one.
            }
        case .bool(let b):
            out.append(b ? "true" : "false")
        case .array(let items):
            out.append("[")
            for (i, item) in items.enumerated() {
                if i > 0 { out.append(",") }
                item.write(into: &out)
            }
            out.append("]")
        case .object(let pairs):
            out.append("{")
            for (i, pair) in pairs.enumerated() {
                if i > 0 { out.append(",") }
                out.append("\"")
                out.append(Self.escape(pair.0))
                out.append("\":")
                pair.1.write(into: &out)
            }
            out.append("}")
        }
    }

    private static func escape(_ s: String) -> String {
        var out = ""
        out.reserveCapacity(s.count)
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": out.append("\\\"")
            case "\\": out.append("\\\\")
            case "\n": out.append("\\n")
            case "\r": out.append("\\r")
            case "\t": out.append("\\t")
            default:
                if scalar.value < 0x20 {
                    out.append(String(format: "\\u%04x", scalar.value))
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out
    }

    /// Formats a Double the way JSON numbers are conventionally written,
    /// without scientific notation and without trailing garbage digits from
    /// naive `Float -> Double` widening.
    private static func formatDouble(_ d: Double) -> String {
        if d == d.rounded(), abs(d) < 1e15 {
            return String(Int64(d))
        }
        // %.9g is enough round-trip precision for a Float32 source value
        // while staying compact and free of exponent notation for the
        // magnitudes splat data actually uses.
        return String(format: "%.9g", d)
    }
}
