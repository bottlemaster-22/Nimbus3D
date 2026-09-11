//
//  BoosterHTTP.swift
//  Booster
//
//  A thin, shared HTTP helper. Every network call in this module - pairing,
//  job control, chunk upload, chunk download - goes through here so headers
//  (API version, auth token), timeouts, JSON coding and error mapping are
//  handled exactly once.
//

import Foundation

/// Everything needed to reach one Booster: its resolved address plus, once
/// paired, its auth token.
public struct BoosterEndpointAddress: Sendable {
    public var host: String
    public var port: UInt16
    public var token: String?

    public init(host: String, port: UInt16, token: String? = nil) {
        self.host = host
        self.port = port
        self.token = token
    }

    public func url(path: String, query: [URLQueryItem] = []) -> URL? {
        var components = URLComponents()
        components.scheme = "http"
        // `percentEncodedHost`, not `host`, for the same reason the path
        // below uses the encoded setter. `BoosterDiscovery.hostString`
        // already produced a URL-shaped host: an IPv6 address arrives
        // bracketed with its zone written %25. The plain `host` setter
        // treats its input as NOT yet encoded, so it would turn that %25
        // into %2525 and the address would be wrong in a new way.
        //
        // For an IPv4 address or a Bonjour name, which are plain ASCII with
        // nothing to escape, the two setters are identical.
        components.percentEncodedHost = host
        components.port = Int(port)
        // `percentEncodedPath`, not `path`: every caller in BoosterAPI that
        // embeds a relative file path has already percent-encoded it with
        // `.urlPathAllowed` (per docs/BOOSTER_PROTOCOL.md section 5, so `/`
        // survives as real path segments). URLComponents.path's setter treats
        // its input as NOT yet encoded and would percent-encode it again on
        // composition - the `%` from a prior encoding pass becomes `%25` and
        // every escaped byte doubles up. percentEncodedPath is composed
        // as-is. Every fixed path segment BoosterAPI builds (the /v1 prefix,
        // job/request ids) is plain ASCII with nothing that needs escaping,
        // so this is safe for the whole composed string, not just the
        // relative-path suffix.
        components.percentEncodedPath = path
        if !query.isEmpty { components.queryItems = query }
        return components.url
    }
}

struct BoosterHTTP {

    static let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.waitsForConnectivity = false
        configuration.timeoutIntervalForRequest = BoosterAPI.requestTimeout
        return URLSession(configuration: configuration)
    }()

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    private static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    /// A JSON request/response round trip with no body.
    static func get<Response: Decodable>(
        _ address: BoosterEndpointAddress,
        path: String,
        query: [URLQueryItem] = []
    ) async throws -> Response {
        var request = try makeRequest(address, path: path, query: query)
        request.httpMethod = "GET"
        return try await send(request)
    }

    /// A JSON request/response round trip with a JSON body.
    static func post<Body: Encodable, Response: Decodable>(
        _ address: BoosterEndpointAddress,
        path: String,
        body: Body
    ) async throws -> Response {
        var request = try makeRequest(address, path: path)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(body)
        return try await send(request)
    }

    /// A POST with no request body, JSON response.
    static func post<Response: Decodable>(
        _ address: BoosterEndpointAddress,
        path: String
    ) async throws -> Response {
        var request = try makeRequest(address, path: path)
        request.httpMethod = "POST"
        return try await send(request)
    }

    static func delete(_ address: BoosterEndpointAddress, path: String) async throws {
        var request = try makeRequest(address, path: path)
        request.httpMethod = "DELETE"
        _ = try await sendRaw(request)
    }

    /// Uploads one raw chunk at a byte offset. Returns the server's
    /// authoritative "bytes received so far" for this file.
    static func putChunk(
        _ address: BoosterEndpointAddress,
        path: String,
        offset: Int64,
        sha256: String,
        data: Data
    ) async throws -> BoosterChunkUploadResponse {
        var request = try makeRequest(address, path: path)
        request.httpMethod = "PUT"
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        request.setValue(String(offset), forHTTPHeaderField: "X-Nimbus-Chunk-Offset")
        request.setValue(sha256, forHTTPHeaderField: "X-Nimbus-Chunk-Sha256")
        request.timeoutInterval = BoosterAPI.uploadChunkTimeout
        request.httpBody = data
        return try await send(request)
    }

    /// Downloads a byte range of a result file. `range` is nil to request the
    /// remainder of the file from `offset` onward with no upper bound.
    static func getRange(
        _ address: BoosterEndpointAddress,
        path: String,
        offset: Int64,
        length: Int?
    ) async throws -> (data: Data, isLastChunk: Bool) {
        var request = try makeRequest(address, path: path)
        request.httpMethod = "GET"
        request.timeoutInterval = BoosterAPI.downloadChunkTimeout
        let upperBound: String
        if let length {
            upperBound = String(offset + Int64(length) - 1)
        } else {
            upperBound = ""
        }
        request.setValue(
            "bytes=\(offset)-\(upperBound)",
            forHTTPHeaderField: "Range"
        )
        let (data, response) = try await sendRaw(request)
        guard let http = response as? HTTPURLResponse else {
            throw BoosterError.connectionFailed("No response from that computer.")
        }
        // A non-2xx range response (job gone, server error, ...) must never
        // be mistaken for file bytes.
        guard http.statusCode == 200 || http.statusCode == 206 else {
            try Self.checkStatus(http, data: data)
            throw BoosterError.resultDownloadFailed(
                "The Booster returned an unexpected error (HTTP \(http.statusCode))."
            )
        }
        // 206 Partial Content = more may follow; 200 = the server does not
        // support ranges and just sent everything, which we treat as final.
        let isLast = http.statusCode == 200 || data.count < (length ?? data.count)
        return (data, isLast)
    }

    // MARK: - Private

    /// PUT one whole file and ignore the reply body.
///
    /// Separate from `putChunk` on purpose: that one carries an offset and
    /// a checksum because a scan upload must survive being interrupted.
    /// This is for development diagnostics, where a failure costs a retry
    /// rather than a scan, so it is one request with no resume machinery.
    static func putFile(
        _ address: BoosterEndpointAddress,
        path: String,
        data: Data
    ) async throws {
        var request = try makeRequest(address, path: path)
        request.httpMethod = "PUT"
        request.setValue(
            "application/octet-stream", forHTTPHeaderField: "Content-Type"
        )
        request.timeoutInterval = BoosterAPI.uploadChunkTimeout
        request.httpBody = data
        let (body, response) = try await sendRaw(request)
        guard let http = response as? HTTPURLResponse else {
            throw BoosterError.connectionFailed("No reply from that computer.")
        }
        try checkStatus(http, data: body)
    }

    private static func makeRequest(
        _ address: BoosterEndpointAddress,
        path: String,
        query: [URLQueryItem] = []
    ) throws -> URLRequest {
        guard let url = address.url(path: path, query: query) else {
            throw BoosterError.connectionFailed("That computer's address looks invalid.")
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = BoosterAPI.requestTimeout
        request.setValue(BoosterAPI.apiVersion, forHTTPHeaderField: "X-Nimbus-Api-Version")
        if let token = address.token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    private static func send<Response: Decodable>(_ request: URLRequest) async throws
        -> Response
    {
        let (data, response) = try await sendRaw(request)
        guard let http = response as? HTTPURLResponse else {
            throw BoosterError.connectionFailed("No response from that computer.")
        }
        try Self.checkStatus(http, data: data)
        do {
            return try decoder.decode(Response.self, from: data)
        } catch {
            throw BoosterError.decodingFailed(error.localizedDescription)
        }
    }

    private static func sendRaw(_ request: URLRequest) async throws -> (Data, URLResponse) {
        do {
            return try await session.data(for: request)
        } catch let error as BoosterError {
            throw error
        } catch {
            let nsError = error as NSError
            if nsError.domain == NSURLErrorDomain,
                nsError.code == NSURLErrorTimedOut
            {
                throw BoosterError.timedOut
            }
            throw BoosterError.connectionFailed(error.localizedDescription)
        }
    }

    private static func checkStatus(_ response: HTTPURLResponse, data: Data) throws {
        switch response.statusCode {
        case 200..<300:
            return
        case 401, 403:
            throw BoosterError.notPaired
        case 404:
            throw BoosterError.serverRejected("The Booster no longer recognises this job.")
        case 409:
            let reason = (try? decoder.decode(BoosterFinalizeResponse.self, from: data))?
                .reason
            throw BoosterError.serverRejected(
                reason ?? "The Booster could not accept this."
            )
        case 426:
            // The Booster is expected to echo its own X-Nimbus-Api-Version on
            // this response so the message can name the actual mismatch
            // instead of a generic "unknown" - fall back only if it didn't.
            let boosterVersion =
                response.value(forHTTPHeaderField: "X-Nimbus-Api-Version") ?? "unknown"
            throw BoosterError.apiVersionMismatch(boosterVersion: boosterVersion)
        default:
            throw BoosterError.serverRejected(
                "The Booster returned an unexpected error (HTTP \(response.statusCode))."
            )
        }
    }
}
