#include <openssl/evp.h>
#include <openssl/rand.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <algorithm> // for std::fill

class SeedCommitter {
public:
    SeedCommitter() = default;
    ~SeedCommitter() = default;

    // Generate unbiased seed for given context
    std::vector<unsigned char> generate(const std::string& trainingID, int t, int batchID) {
        std::string context = trainingID + "|" + std::to_string(t) + "|" + std::to_string(batchID);

        size_t seedSize = EVP_MD_size(EVP_sha256());
        std::vector<unsigned char> k = generateRandom(seedSize);

        std::vector<unsigned char> c = commit(k, context);
        std::vector<unsigned char> r = hashSHA256(c);

        if (r.size() != k.size()) r.resize(k.size(), 0);

        std::vector<unsigned char> s = xorVec(k, r);

        // Zero-out sensitive data
        secureZero(k);
        secureZero(r);

        return s;
    }

    // Generate seed and return as hex string
    std::string generateHex(const std::string& trainingID, int t, int batchID) {
        std::vector<unsigned char> seed = generate(trainingID, t, batchID);
        return toHex(seed);
    }

private:
    // Hash function using EVP SHA-256
    std::vector<unsigned char> hashSHA256(const std::vector<unsigned char>& input) {
        const EVP_MD* md = EVP_sha256();
        if (!md) throw std::runtime_error("EVP_sha256() failed");

        EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        if (!ctx) throw std::runtime_error("EVP_MD_CTX_new failed");

        if (EVP_DigestInit_ex(ctx, md, nullptr) != 1)
            throw std::runtime_error("EVP_DigestInit_ex failed");
        if (EVP_DigestUpdate(ctx, input.data(), input.size()) != 1)
            throw std::runtime_error("EVP_DigestUpdate failed");

        unsigned int outLen = EVP_MD_size(md);
        std::vector<unsigned char> out(outLen);
        if (EVP_DigestFinal_ex(ctx, out.data(), &outLen) != 1)
            throw std::runtime_error("EVP_DigestFinal_ex failed");

        EVP_MD_CTX_free(ctx);
        out.resize(outLen);
        return out;
    }

    // Commitment: hash(k || context)
    std::vector<unsigned char> commit(const std::vector<unsigned char>& k, const std::string& context) {
        std::vector<unsigned char> buf = k;
        buf.insert(buf.end(), context.begin(), context.end());
        return hashSHA256(buf);
    }

    // CSPRNG using OpenSSL
    std::vector<unsigned char> generateRandom(size_t length) {
        std::vector<unsigned char> buf(length);
        if (RAND_bytes(buf.data(), (int)length) != 1)
            throw std::runtime_error("RAND_bytes failed");
        return buf;
    }

    // XOR two vectors
    std::vector<unsigned char> xorVec(const std::vector<unsigned char>& a,
                                      const std::vector<unsigned char>& b) {
        if (a.size() != b.size()) throw std::runtime_error("xor size mismatch");
        std::vector<unsigned char> out(a.size());
        for (size_t i = 0; i < a.size(); i++) out[i] = a[i] ^ b[i];
        return out;
    }

    // Convert bytes to hex string
    std::string toHex(const std::vector<unsigned char>& v) {
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (unsigned char b : v) oss << std::setw(2) << (int)b;
        return oss.str();
    }

    // Securely zero memory
    void secureZero(std::vector<unsigned char>& v) {
        std::fill(v.begin(), v.end(), 0);
    }
};

// Example usage
int main() {
    try {
        SeedCommitter committer;
        std::string hexSeed = committer.generateHex("trainingABC", 100, 5);
        std::cout << "Seed (hex): " << hexSeed << std::endl;

        // Raw byte vector if needed
        std::vector<unsigned char> rawSeed = committer.generate("trainingABC", 100, 5);
        std::cout << "Seed size (bytes): " << rawSeed.size() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
