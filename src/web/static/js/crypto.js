/**
 * Agent OS Encryption Utilities
 *
 * Provides client-side encryption for sensitive data using Web Crypto API.
 * Uses AES-256-GCM for symmetric encryption with PBKDF2 key derivation.
 */

class SecureCrypto {
    constructor() {
        // Encryption configuration
        this.algorithm = 'AES-GCM';
        this.keyLength = 256;
        this.ivLength = 12;
        this.saltLength = 16;
        this.iterations = 100000;

        // Key cache (session only - not persisted)
        this._keyCache = null;
        this._keyCacheTime = null;
        this._keyCacheMaxAge = 30 * 60 * 1000; // 30 minutes

        // Check for Web Crypto API support
        this.isSupported = !!(window.crypto && window.crypto.subtle);

        if (!this.isSupported) {
            console.warn('Web Crypto API not supported - falling back to obfuscation');
        }
    }

    /**
     * Generate a cryptographically secure random string
     */
    generateRandomBytes(length) {
        const array = new Uint8Array(length);
        window.crypto.getRandomValues(array);
        return array;
    }

    /**
     * Convert ArrayBuffer to Base64 string
     */
    arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }

    /**
     * Convert Base64 string to ArrayBuffer
     */
    base64ToArrayBuffer(base64) {
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        return bytes.buffer;
    }

    /**
     * Get or derive encryption key
     * Uses a combination of browser fingerprint and stored salt for key derivation
     */
    async getEncryptionKey() {
        // Check cache
        if (this._keyCache && this._keyCacheTime) {
            const age = Date.now() - this._keyCacheTime;
            if (age < this._keyCacheMaxAge) {
                return this._keyCache;
            }
        }

        // Get or generate device-specific salt
        let storedSalt = localStorage.getItem('_aos_ks');
        let salt;

        if (storedSalt) {
            salt = new Uint8Array(this.base64ToArrayBuffer(storedSalt));
        } else {
            salt = this.generateRandomBytes(this.saltLength);
            localStorage.setItem('_aos_ks', this.arrayBufferToBase64(salt));
        }

        // Create a passphrase from browser characteristics (for additional entropy)
        const fingerprint = this._generateFingerprint();

        // Import the passphrase as key material
        const keyMaterial = await window.crypto.subtle.importKey(
            'raw',
            new TextEncoder().encode(fingerprint),
            'PBKDF2',
            false,
            ['deriveKey']
        );

        // Derive the actual encryption key
        const key = await window.crypto.subtle.deriveKey(
            {
                name: 'PBKDF2',
                salt: salt,
                iterations: this.iterations,
                hash: 'SHA-256'
            },
            keyMaterial,
            { name: this.algorithm, length: this.keyLength },
            false,
            ['encrypt', 'decrypt']
        );

        // Cache the key
        this._keyCache = key;
        this._keyCacheTime = Date.now();

        return key;
    }

    /**
     * Generate browser fingerprint for key derivation
     * Not for tracking - just to add device-specific entropy
     */
    _generateFingerprint() {
        const components = [
            navigator.userAgent,
            navigator.language,
            screen.colorDepth,
            screen.width + 'x' + screen.height,
            new Date().getTimezoneOffset(),
            navigator.hardwareConcurrency || 0,
            'aos-secure-v1'
        ];
        return components.join('|');
    }

    /**
     * Encrypt a string value
     * @param {string} plaintext - The value to encrypt
     * @returns {Promise<string>} - Base64 encoded encrypted value with IV prefix
     */
    async encrypt(plaintext) {
        if (!this.isSupported) {
            return this._obfuscate(plaintext);
        }

        try {
            const key = await this.getEncryptionKey();
            const iv = this.generateRandomBytes(this.ivLength);
            const encodedText = new TextEncoder().encode(plaintext);

            const ciphertext = await window.crypto.subtle.encrypt(
                { name: this.algorithm, iv: iv },
                key,
                encodedText
            );

            // Combine IV + ciphertext and encode as base64
            const combined = new Uint8Array(iv.length + ciphertext.byteLength);
            combined.set(iv, 0);
            combined.set(new Uint8Array(ciphertext), iv.length);

            return 'enc:' + this.arrayBufferToBase64(combined);
        } catch (error) {
            console.error('Encryption error:', error);
            // Fall back to obfuscation
            return this._obfuscate(plaintext);
        }
    }

    /**
     * Decrypt an encrypted string value
     * @param {string} encryptedValue - Base64 encoded encrypted value with IV prefix
     * @returns {Promise<string>} - Decrypted plaintext
     */
    async decrypt(encryptedValue) {
        // Check if value is encrypted
        if (!encryptedValue.startsWith('enc:')) {
            // Check for obfuscated value
            if (encryptedValue.startsWith('obs:')) {
                return this._deobfuscate(encryptedValue);
            }
            // Plain value - return as-is (backward compatibility)
            return encryptedValue;
        }

        if (!this.isSupported) {
            console.warn('Cannot decrypt - Web Crypto not supported');
            return encryptedValue;
        }

        try {
            const key = await this.getEncryptionKey();
            const combined = new Uint8Array(this.base64ToArrayBuffer(encryptedValue.slice(4)));

            // Extract IV and ciphertext
            const iv = combined.slice(0, this.ivLength);
            const ciphertext = combined.slice(this.ivLength);

            const plaintext = await window.crypto.subtle.decrypt(
                { name: this.algorithm, iv: iv },
                key,
                ciphertext
            );

            return new TextDecoder().decode(plaintext);
        } catch (error) {
            console.error('Decryption error:', error);
            return encryptedValue;
        }
    }

    /**
     * Simple obfuscation fallback for browsers without Web Crypto
     */
    _obfuscate(text) {
        const encoded = btoa(unescape(encodeURIComponent(text)));
        // Simple XOR with rotating key
        const key = 'aos-secure';
        let result = '';
        for (let i = 0; i < encoded.length; i++) {
            result += String.fromCharCode(encoded.charCodeAt(i) ^ key.charCodeAt(i % key.length));
        }
        return 'obs:' + btoa(result);
    }

    /**
     * Deobfuscate value
     */
    _deobfuscate(obfuscated) {
        try {
            const encoded = atob(obfuscated.slice(4));
            const key = 'aos-secure';
            let result = '';
            for (let i = 0; i < encoded.length; i++) {
                result += String.fromCharCode(encoded.charCodeAt(i) ^ key.charCodeAt(i % key.length));
            }
            return decodeURIComponent(escape(atob(result)));
        } catch {
            return obfuscated;
        }
    }

    /**
     * Clear cached key (call on logout)
     */
    clearKeyCache() {
        this._keyCache = null;
        this._keyCacheTime = null;
    }
}


/**
 * Secure Storage
 *
 * Wrapper around localStorage that encrypts sensitive values.
 */
class SecureStorage {
    constructor(crypto) {
        this.crypto = crypto || new SecureCrypto();

        // Keys that should be encrypted
        this.sensitiveKeys = new Set([
            'ollamaEndpoint',
            'llamaCppEndpoint',
            'openaiApiKey',
            'anthropicApiKey',
            'huggingfaceToken',
            'stableDiffusionEndpoint',
            'comfyuiEndpoint',
            'automaticEndpoint',
            'dbConnectionString',
            'authToken',
            'refreshToken',
            'apiKeys',
            'credentials',
            'password',
            'secret'
        ]);

        // Pattern matchers for sensitive keys
        this.sensitivePatterns = [
            /api[_-]?key/i,
            /token/i,
            /secret/i,
            /password/i,
            /credential/i,
            /endpoint/i,
            /connection/i
        ];
    }

    /**
     * Check if a key should be encrypted
     */
    isSensitive(key) {
        if (this.sensitiveKeys.has(key)) {
            return true;
        }

        for (const pattern of this.sensitivePatterns) {
            if (pattern.test(key)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Store a value securely
     */
    async setItem(key, value) {
        if (this.isSensitive(key) && value) {
            const encrypted = await this.crypto.encrypt(String(value));
            localStorage.setItem(key, encrypted);
        } else {
            localStorage.setItem(key, value);
        }
    }

    /**
     * Retrieve a value (auto-decrypts if encrypted)
     */
    async getItem(key) {
        const value = localStorage.getItem(key);
        if (!value) return value;

        // Check if value is encrypted
        if (value.startsWith('enc:') || value.startsWith('obs:')) {
            return await this.crypto.decrypt(value);
        }

        // If key is sensitive but value isn't encrypted, encrypt it now (migration)
        if (this.isSensitive(key)) {
            const encrypted = await this.crypto.encrypt(value);
            localStorage.setItem(key, encrypted);
        }

        return value;
    }

    /**
     * Remove an item
     */
    removeItem(key) {
        localStorage.removeItem(key);
    }

    /**
     * Clear all storage
     */
    clear() {
        localStorage.clear();
        this.crypto.clearKeyCache();
    }

    /**
     * Migrate existing plain values to encrypted
     */
    async migrateToEncrypted() {
        const keys = Object.keys(localStorage);
        let migrated = 0;

        for (const key of keys) {
            if (this.isSensitive(key)) {
                const value = localStorage.getItem(key);
                if (value && !value.startsWith('enc:') && !value.startsWith('obs:')) {
                    const encrypted = await this.crypto.encrypt(value);
                    localStorage.setItem(key, encrypted);
                    migrated++;
                }
            }
        }

        if (migrated > 0) {
            console.log(`Migrated ${migrated} sensitive values to encrypted storage`);
        }

        return migrated;
    }
}


/**
 * Sensitive Data Redactor
 *
 * Redacts sensitive information from logs and debug output.
 */
class SensitiveDataRedactor {
    constructor() {
        // Patterns to redact (pattern, replacement)
        this.patterns = [
            // API Keys (various formats)
            { regex: /([a-zA-Z0-9_-]*(?:api[_-]?key|apikey)[a-zA-Z0-9_-]*)[=:]\s*["']?([a-zA-Z0-9_-]{20,})["']?/gi, replacement: '$1=[REDACTED]' },
            { regex: /sk-[a-zA-Z0-9]{20,}/g, replacement: '[REDACTED_API_KEY]' },
            { regex: /hf_[a-zA-Z0-9]{20,}/g, replacement: '[REDACTED_HF_TOKEN]' },
            { regex: /ghp_[a-zA-Z0-9]{20,}/g, replacement: '[REDACTED_GH_TOKEN]' },
            { regex: /gho_[a-zA-Z0-9]{20,}/g, replacement: '[REDACTED_GH_TOKEN]' },

            // Bearer tokens
            { regex: /Bearer\s+[a-zA-Z0-9_.-]{20,}/gi, replacement: 'Bearer [REDACTED]' },

            // Basic auth
            { regex: /Basic\s+[a-zA-Z0-9+/=]{20,}/gi, replacement: 'Basic [REDACTED]' },

            // Passwords in URLs
            { regex: /(https?:\/\/[^:]+:)[^@]+(@)/gi, replacement: '$1[REDACTED]$2' },

            // Generic password fields
            { regex: /(password|passwd|pwd)[=:]\s*["']?[^"'\s&]+["']?/gi, replacement: '$1=[REDACTED]' },

            // Secret/token fields
            { regex: /(secret|token)[=:]\s*["']?[a-zA-Z0-9_.-]{10,}["']?/gi, replacement: '$1=[REDACTED]' },

            // Connection strings
            { regex: /(mongodb|postgresql|mysql|redis):\/\/[^:]+:[^@]+@/gi, replacement: '$1://[REDACTED]:[REDACTED]@' },

            // Credit card numbers (basic pattern)
            { regex: /\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b/g, replacement: '[REDACTED_CC]' },

            // Social security numbers (US format)
            { regex: /\b\d{3}-\d{2}-\d{4}\b/g, replacement: '[REDACTED_SSN]' },

            // Email in sensitive context
            { regex: /(email|mail)[=:]\s*["']?[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}["']?/gi, replacement: '$1=[REDACTED_EMAIL]' },

            // JWT tokens
            { regex: /eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*/g, replacement: '[REDACTED_JWT]' },

            // Private keys
            { regex: /-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(RSA\s+)?PRIVATE\s+KEY-----/g, replacement: '[REDACTED_PRIVATE_KEY]' },

            // AWS keys
            { regex: /AKIA[0-9A-Z]{16}/g, replacement: '[REDACTED_AWS_KEY]' },
            { regex: /[a-zA-Z0-9/+=]{40}/g, match: (m) => m.length === 40, replacement: '[POSSIBLE_SECRET]' }
        ];

        // Sensitive field names to fully redact values
        this.sensitiveFields = new Set([
            'password', 'passwd', 'pwd', 'secret', 'token', 'apikey', 'api_key',
            'apiKey', 'auth', 'authorization', 'credential', 'credentials',
            'private_key', 'privateKey', 'access_token', 'accessToken',
            'refresh_token', 'refreshToken', 'session_id', 'sessionId'
        ]);
    }

    /**
     * Redact sensitive data from a string
     */
    redact(text) {
        if (typeof text !== 'string') {
            if (typeof text === 'object') {
                return this.redactObject(text);
            }
            return text;
        }

        let result = text;

        for (const { regex, replacement } of this.patterns) {
            result = result.replace(regex, replacement);
        }

        return result;
    }

    /**
     * Redact sensitive data from an object
     */
    redactObject(obj, depth = 0) {
        if (depth > 10) return obj; // Prevent infinite recursion

        if (obj === null || obj === undefined) {
            return obj;
        }

        if (Array.isArray(obj)) {
            return obj.map(item => this.redactObject(item, depth + 1));
        }

        if (typeof obj !== 'object') {
            if (typeof obj === 'string') {
                return this.redact(obj);
            }
            return obj;
        }

        const result = {};

        for (const [key, value] of Object.entries(obj)) {
            const lowerKey = key.toLowerCase();

            // Check if field name indicates sensitive data
            if (this.sensitiveFields.has(lowerKey) ||
                this.sensitiveFields.has(key) ||
                /password|secret|token|key|credential|auth/i.test(key)) {
                result[key] = '[REDACTED]';
            } else if (typeof value === 'string') {
                result[key] = this.redact(value);
            } else if (typeof value === 'object') {
                result[key] = this.redactObject(value, depth + 1);
            } else {
                result[key] = value;
            }
        }

        return result;
    }

    /**
     * Redact URL parameters
     */
    redactUrl(url) {
        try {
            const parsed = new URL(url);

            // Redact password in URL
            if (parsed.password) {
                parsed.password = '[REDACTED]';
            }

            // Redact sensitive query parameters
            for (const key of parsed.searchParams.keys()) {
                if (/token|key|secret|password|auth|credential/i.test(key)) {
                    parsed.searchParams.set(key, '[REDACTED]');
                }
            }

            return parsed.toString();
        } catch {
            // If URL parsing fails, apply regex redaction
            return this.redact(url);
        }
    }
}


// Export instances for global use
const secureCrypto = new SecureCrypto();
const secureStorage = new SecureStorage(secureCrypto);
const redactor = new SensitiveDataRedactor();

// Make available globally
window.SecureCrypto = SecureCrypto;
window.SecureStorage = SecureStorage;
window.SensitiveDataRedactor = SensitiveDataRedactor;
window.secureCrypto = secureCrypto;
window.secureStorage = secureStorage;
window.redactor = redactor;
