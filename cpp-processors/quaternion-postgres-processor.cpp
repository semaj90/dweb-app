// quaternion-postgres-processor.cpp
// Production C++ processor for quaternion transforms with PostgreSQL integration
// Compile: g++ -std=c++17 quaternion-postgres-processor.cpp -lpqxx -lpq -O3 -o quaternion-postgres-processor.exe

#include <pqxx/pqxx>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <regex>
#include <chrono>
#include <memory>
#include <map>
#include <iomanip>

using Vec3 = Eigen::Vector3f;
using Quat = Eigen::Quaternionf;
using Transform = Eigen::Affine3f;

// Simple JSON parser for JSONB data
class SimpleJSON {
private:
    std::map<std::string, std::string> values;
    std::vector<float> array_values;
    bool is_array = false;
    
public:
    SimpleJSON(const std::string& json_str) {
        parse(json_str);
    }
    
    void parse(const std::string& json_str) {
        if (json_str.empty()) return;
        
        // Check if it's an array
        if (json_str.front() == '[' && json_str.back() == ']') {
            is_array = true;
            parse_array(json_str);
        } else if (json_str.front() == '{' && json_str.back() == '}') {
            parse_object(json_str);
        }
    }
    
    void parse_array(const std::string& json_str) {
        std::string content = json_str.substr(1, json_str.length() - 2);
        std::regex number_regex(R"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)");
        std::sregex_iterator iter(content.begin(), content.end(), number_regex);
        std::sregex_iterator end;
        
        for (; iter != end; ++iter) {
            try {
                float value = std::stof(iter->str());
                array_values.push_back(value);
            } catch (...) {
                // Skip invalid numbers
            }
        }
    }
    
    void parse_object(const std::string& json_str) {
        std::string content = json_str.substr(1, json_str.length() - 2);
        
        // Simple key-value extraction
        std::regex kv_regex(R"("([^"]+)"\s*:\s*([^,}]+))");
        std::sregex_iterator iter(content.begin(), content.end(), kv_regex);
        std::sregex_iterator end;
        
        for (; iter != end; ++iter) {
            std::string key = (*iter)[1].str();
            std::string value = (*iter)[2].str();
            
            // Remove quotes from string values
            if (value.front() == '"' && value.back() == '"') {
                value = value.substr(1, value.length() - 2);
            }
            
            values[key] = value;
        }
    }
    
    float get_float(const std::string& key, float default_val = 0.0f) const {
        auto it = values.find(key);
        if (it != values.end()) {
            try {
                return std::stof(it->second);
            } catch (...) {}
        }
        return default_val;
    }
    
    std::string get_string(const std::string& key, const std::string& default_val = "") const {
        auto it = values.find(key);
        if (it != values.end()) {
            return it->second;
        }
        return default_val;
    }
    
    std::vector<float> get_array() const {
        return array_values;
    }
    
    std::string to_array_json() const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < array_values.size(); ++i) {
            if (i > 0) oss << ",";
            oss << std::fixed << std::setprecision(6) << array_values[i];
        }
        oss << "]";
        return oss.str();
    }
    
    void set_array(const std::vector<float>& arr) {
        array_values = arr;
        is_array = true;
    }
};

class QuaternionProcessor {
private:
    std::unique_ptr<pqxx::connection> conn;
    
public:
    QuaternionProcessor(const std::string& connection_string) {
        try {
            conn = std::make_unique<pqxx::connection>(connection_string);
            if (!conn->is_open()) {
                throw std::runtime_error("Failed to connect to database");
            }
            std::cout << "Connected to PostgreSQL: " << conn->dbname() << std::endl;
        } catch (const std::exception& e) {
            throw std::runtime_error("Database connection failed: " + std::string(e.what()));
        }
    }
    
    Quat parse_quaternion(const SimpleJSON& pose_json) {
        float w = pose_json.get_float("w", 1.0f);
        float x = pose_json.get_float("x", 0.0f);
        float y = pose_json.get_float("y", 0.0f);
        float z = pose_json.get_float("z", 0.0f);
        
        Quat q(w, x, y, z);
        q.normalize();
        return q;
    }
    
    Vec3 parse_translation(const SimpleJSON& pose_json) {
        float tx = pose_json.get_float("tx", 0.0f);
        float ty = pose_json.get_float("ty", 0.0f);
        float tz = pose_json.get_float("tz", 0.0f);
        return Vec3(tx, ty, tz);
    }
    
    std::vector<Vec3> rotate_point_cloud(const std::vector<float>& flat_points, const Quat& rotation) {
        std::vector<Vec3> rotated_points;
        rotated_points.reserve(flat_points.size() / 3);
        
        for (size_t i = 0; i + 2 < flat_points.size(); i += 3) {
            Vec3 point(flat_points[i], flat_points[i + 1], flat_points[i + 2]);
            Vec3 rotated = rotation * point;
            rotated_points.push_back(rotated);
        }
        
        return rotated_points;
    }
    
    std::vector<Vec3> transform_point_cloud(const std::vector<float>& flat_points, 
                                           const Quat& rotation, const Vec3& translation) {
        std::vector<Vec3> transformed_points;
        transformed_points.reserve(flat_points.size() / 3);
        
        Transform transform = Transform::Identity();
        transform.linear() = rotation.toRotationMatrix();
        transform.translation() = translation;
        
        for (size_t i = 0; i + 2 < flat_points.size(); i += 3) {
            Vec3 point(flat_points[i], flat_points[i + 1], flat_points[i + 2]);
            Vec3 transformed = transform * point;
            transformed_points.push_back(transformed);
        }
        
        return transformed_points;
    }
    
    std::vector<float> vec3_to_flat(const std::vector<Vec3>& points) {
        std::vector<float> flat;
        flat.reserve(points.size() * 3);
        
        for (const auto& p : points) {
            flat.push_back(p.x());
            flat.push_back(p.y());
            flat.push_back(p.z());
        }
        
        return flat;
    }
    
    int process_chunks_batch(int batch_size = 100) {
        try {
            pqxx::work txn(*conn);
            
            // Select chunks with point clouds that need processing
            std::string query = R"(
                SELECT id, point_cloud, pose, meta 
                FROM chunks 
                WHERE point_cloud IS NOT NULL 
                  AND (rotated_points IS NULL OR jsonb_array_length(rotated_points) = 0)
                LIMIT $1
            )";
            
            pqxx::result result = txn.exec_params(query, batch_size);
            
            if (result.empty()) {
                std::cout << "No chunks found for processing" << std::endl;
                return 0;
            }
            
            int processed_count = 0;
            auto start_time = std::chrono::high_resolution_clock::now();
            
            for (const auto& row : result) {
                try {
                    std::string chunk_id = row["id"].as<std::string>();
                    std::string point_cloud_json = row["point_cloud"].as<std::string>();
                    std::string pose_json = row["pose"].is_null() ? "{\"w\":1,\"x\":0,\"y\":0,\"z\":0}" : row["pose"].as<std::string>();
                    std::string meta_json = row["meta"].is_null() ? "{}" : row["meta"].as<std::string>();
                    
                    // Parse input data
                    SimpleJSON points_parser(point_cloud_json);
                    SimpleJSON pose_parser(pose_json);
                    SimpleJSON meta_parser(meta_json);
                    
                    std::vector<float> input_points = points_parser.get_array();
                    
                    if (input_points.empty() || input_points.size() % 3 != 0) {
                        std::cerr << "Invalid point cloud for chunk " << chunk_id << std::endl;
                        continue;
                    }
                    
                    // Extract quaternion and translation
                    Quat rotation = parse_quaternion(pose_parser);
                    Vec3 translation = parse_translation(pose_parser);
                    
                    // Process points
                    std::vector<Vec3> rotated_points;
                    std::string processing_type = meta_parser.get_string("processing_type", "rotation");
                    
                    if (processing_type == "transform") {
                        rotated_points = transform_point_cloud(input_points, rotation, translation);
                    } else {
                        rotated_points = rotate_point_cloud(input_points, rotation);
                    }
                    
                    // Convert back to flat array
                    std::vector<float> output_flat = vec3_to_flat(rotated_points);
                    SimpleJSON output_json("");
                    output_json.set_array(output_flat);
                    
                    // Update database
                    std::string update_query = R"(
                        UPDATE chunks 
                        SET rotated_points = $1::jsonb,
                            updated_at = NOW(),
                            meta = jsonb_set(
                                COALESCE(meta, '{}'), 
                                '{processing_stats}', 
                                jsonb_build_object(
                                    'processed_at', NOW()::text,
                                    'input_points', $2,
                                    'output_points', $3,
                                    'processing_type', $4
                                )
                            )
                        WHERE id = $5
                    )";
                    
                    txn.exec_params(update_query, 
                                   output_json.to_array_json(),
                                   static_cast<int>(input_points.size() / 3),
                                   static_cast<int>(output_flat.size() / 3),
                                   processing_type,
                                   chunk_id);
                    
                    processed_count++;
                    
                    if (processed_count % 10 == 0) {
                        std::cout << "Processed " << processed_count << " chunks..." << std::endl;
                    }
                    
                } catch (const std::exception& e) {
                    std::cerr << "Error processing chunk: " << e.what() << std::endl;
                    continue;
                }
            }
            
            txn.commit();
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            std::cout << "Successfully processed " << processed_count 
                      << " chunks in " << duration.count() << " ms" << std::endl;
            
            return processed_count;
            
        } catch (const std::exception& e) {
            std::cerr << "Batch processing error: " << e.what() << std::endl;
            return -1;
        }
    }
    
    bool process_single_chunk(const std::string& chunk_id) {
        try {
            pqxx::work txn(*conn);
            
            std::string query = R"(
                SELECT point_cloud, pose, meta 
                FROM chunks 
                WHERE id = $1 AND point_cloud IS NOT NULL
            )";
            
            pqxx::result result = txn.exec_params(query, chunk_id);
            
            if (result.empty()) {
                std::cerr << "Chunk not found or has no point cloud: " << chunk_id << std::endl;
                return false;
            }
            
            auto row = result[0];
            std::string point_cloud_json = row["point_cloud"].as<std::string>();
            std::string pose_json = row["pose"].is_null() ? "{\"w\":1,\"x\":0,\"y\":0,\"z\":0}" : row["pose"].as<std::string>();
            
            SimpleJSON points_parser(point_cloud_json);
            SimpleJSON pose_parser(pose_json);
            
            std::vector<float> input_points = points_parser.get_array();
            Quat rotation = parse_quaternion(pose_parser);
            Vec3 translation = parse_translation(pose_parser);
            
            std::vector<Vec3> rotated_points = rotate_point_cloud(input_points, rotation);
            std::vector<float> output_flat = vec3_to_flat(rotated_points);
            
            SimpleJSON output_json("");
            output_json.set_array(output_flat);
            
            std::string update_query = R"(
                UPDATE chunks 
                SET rotated_points = $1::jsonb,
                    updated_at = NOW()
                WHERE id = $2
            )";
            
            txn.exec_params(update_query, output_json.to_array_json(), chunk_id);
            txn.commit();
            
            std::cout << "Successfully processed chunk: " << chunk_id << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error processing chunk " << chunk_id << ": " << e.what() << std::endl;
            return false;
        }
    }
    
    void create_required_columns() {
        try {
            pqxx::work txn(*conn);
            
            // Add rotated_points column if it doesn't exist
            std::string alter_query = R"(
                DO $$ BEGIN
                    BEGIN
                        ALTER TABLE chunks ADD COLUMN rotated_points JSONB;
                        ALTER TABLE chunks ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
                    EXCEPTION
                        WHEN duplicate_column THEN 
                        -- Column already exists, do nothing
                    END;
                END $$;
            )";
            
            txn.exec(alter_query);
            txn.commit();
            
            std::cout << "Database schema updated successfully" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Error updating schema: " << e.what() << std::endl;
        }
    }
    
    void print_stats() {
        try {
            pqxx::work txn(*conn);
            
            pqxx::result stats = txn.exec(R"(
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(point_cloud) as chunks_with_points,
                    COUNT(rotated_points) as processed_chunks,
                    COUNT(CASE WHEN point_cloud IS NOT NULL AND rotated_points IS NULL THEN 1 END) as pending_chunks
                FROM chunks
            )");
            
            if (!stats.empty()) {
                auto row = stats[0];
                std::cout << "\n=== Processing Statistics ===" << std::endl;
                std::cout << "Total chunks: " << row["total_chunks"].as<int>() << std::endl;
                std::cout << "Chunks with point clouds: " << row["chunks_with_points"].as<int>() << std::endl;
                std::cout << "Processed chunks: " << row["processed_chunks"].as<int>() << std::endl;
                std::cout << "Pending chunks: " << row["pending_chunks"].as<int>() << std::endl;
                std::cout << "==============================\n" << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error getting stats: " << e.what() << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    try {
        // Get database connection string
        const char* db_url = std::getenv("DATABASE_URL");
        std::string connection_string = db_url ? std::string(db_url) : 
            "postgresql://legal_admin:123456@localhost:5432/legal_ai_db";
        
        QuaternionProcessor processor(connection_string);
        
        // Parse command line arguments
        std::string command = argc > 1 ? argv[1] : "batch";
        
        if (command == "init") {
            std::cout << "Initializing database schema..." << std::endl;
            processor.create_required_columns();
            
        } else if (command == "stats") {
            processor.print_stats();
            
        } else if (command == "single" && argc > 2) {
            std::string chunk_id = argv[2];
            std::cout << "Processing single chunk: " << chunk_id << std::endl;
            processor.process_single_chunk(chunk_id);
            
        } else if (command == "batch") {
            int batch_size = argc > 2 ? std::stoi(argv[2]) : 100;
            std::cout << "Starting batch processing (batch size: " << batch_size << ")..." << std::endl;
            
            processor.print_stats();
            int processed = processor.process_chunks_batch(batch_size);
            
            if (processed > 0) {
                std::cout << "\nFinal statistics:" << std::endl;
                processor.print_stats();
            }
            
        } else {
            std::cout << "Usage: " << argv[0] << " [command] [options]" << std::endl;
            std::cout << "Commands:" << std::endl;
            std::cout << "  init              - Initialize database schema" << std::endl;
            std::cout << "  stats             - Show processing statistics" << std::endl;
            std::cout << "  single <chunk_id> - Process a single chunk" << std::endl;
            std::cout << "  batch [size]      - Process chunks in batches (default: 100)" << std::endl;
            return 1;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}