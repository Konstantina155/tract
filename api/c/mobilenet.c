#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <sys/time.h>  // for gettimeofday()
#include <tract.h>

#define check(call) {                                                           \
    TRACT_RESULT result = call;                                                 \
    if(result == TRACT_RESULT_KO) {                                             \
        fprintf(stderr, "Error calling tract: %s", tract_get_last_error());     \
        exit(1) ;                                                               \
    }                                                                           \
}

typedef struct prediction {
    double pred;
    int category;
    TractValue *output;
    TractInferenceModel *inference_model;
}prediction;

prediction **
init_predictions(int arg)
{
    prediction **inf = malloc(arg * sizeof(prediction));
    if (!inf) {
        fprintf(stderr, "Error allocating memory for predictions\n");
        return NULL;
    }

    for (int i = 0; i < arg; i++) {
        inf[i] = malloc(sizeof(prediction));
        if (!inf[i]) {
            fprintf(stderr, "Error allocating memory for prediction\n");
            return NULL;
        }
        inf[i]->pred = 0.0;
        inf[i]->category = 0;
        inf[i]->output = NULL;
        inf[i]->inference_model = NULL;
    }
    return inf;
}

void
free_prediction(prediction *inf)
{
    if (!inf) {
        return;
    }
    if (inf->output) {
        check(tract_value_destroy(&inf->output));
        assert(!inf->output);
    }
    if (inf->inference_model) {
        fprintf(stderr, "Destroying inference model\n");
        check(tract_inference_model_release(&inf->inference_model));
        assert(!inf->inference_model);
    }
    free(inf);
}

void
free_predictions(prediction **inf, int length)
{
    for (int i = 0; i < length; i++) {
        free_prediction(inf[i]);
    }
    free(inf);
}

prediction *
inference(char *model_name, TractValue *input, TractValue *input2, prediction *inf, struct EncryptionParameters *params)
{
    struct timeval t1_inf, t2_inf;
    double elapsed_time;

    // Initialize onnx parser
    TractOnnx *onnx = NULL;
    check(tract_onnx_create(&onnx));
    assert(onnx);

    // Load the model
    TractModel *model = NULL;
    TractInferenceModel *inference_model = NULL;
    if (tract_onnx_model_for_path(onnx, model_name, &inference_model, params) != TRACT_RESULT_OK) {
        fprintf(stderr, "Error loading model %s\n", model_name);
        free_prediction(inf);
        check(tract_onnx_destroy(&onnx));
        assert(!onnx);
        return NULL;
    }
    assert(inference_model);
    assert(onnx);

    check(tract_onnx_destroy(&onnx));
    assert(!onnx);

    // Convert inference model to a typed model and optimize it
    check(tract_inference_model_into_typed(&inference_model,&model));
    assert(model);

    // Make the model runnable
    TractRunnable *runnable = NULL;
    check(tract_model_into_runnable(&model, &runnable));
    assert(runnable);
    assert(!model);

    TractValue* output = NULL;

    gettimeofday(&t1_inf, NULL);

    // simple stateless run...
    TractValue *inputs[] = { input, input2 };
    fprintf(stderr, "Run model %s\n", model_name);
    check(tract_runnable_run(runnable, inputs, &output));

    const float *data = NULL;
    check(tract_value_as_bytes(output, NULL, NULL, NULL, (const void**) &data));

    check(tract_runnable_release(&runnable));
    assert(!runnable);

    float max = data[0];
    int argmax = 0;
    for(int i = 0; i < 1000; i++) {
        float val = data[i];
        if(val > max) {
            max = val;
            argmax = i;
        }
    }
    assert(data[argmax] == max);
    fprintf(stderr, "\nModel: %s\nMax is %f for category %d\n", model_name, max, argmax);

    
    gettimeofday(&t2_inf, NULL);
    elapsed_time = (t2_inf.tv_sec - t1_inf.tv_sec) * 1000.0;      // sec to ms
    elapsed_time += (t2_inf.tv_usec - t1_inf.tv_usec) / 1000.0;   // us to ms
    printf("Model run in %f ms.\n", elapsed_time);

    inf->output = output;
    inf->pred = max;
    inf->category = argmax;
    inf->inference_model = inference_model;

    return inf;
}

size_t *
decode_pb(FILE *fd)
{
    uint8_t byte;
    uint8_t wire_type;
    uint64_t varint_value;
    static size_t shape[4] = {0, 0, 0, 0};

    int k=0;
    for (int k = 0; k < 4; k++) {
        fread(&byte, sizeof(uint8_t), 1, fd);
        wire_type = byte & 0x07;

        if (wire_type == 0) {
            varint_value = 0;
            int shift = 0;
            do {
                fread(&byte, sizeof(uint8_t), 1, fd);
                varint_value |= (uint64_t)(byte & 0x7F) << (7 * shift);
                shift++;
            } while (byte & 0x80);
            shape[k] = varint_value;
        } else {
            break;
        }
    }

    fseek(fd, 0, SEEK_SET);

    return shape;
}

// test tract decryption
#define KEY_BYTES 32
#define IV_BYTES 12
#define TAG_BYTES 16
#define ADD_DATA_BYTES 64

uint8_t *
write_to_buffer(char *filename)
{
    FILE *fd = fopen(filename, "rb");
    if (!fd) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return NULL;
    }
    fseek(fd, 0, SEEK_END);
    long file_size = ftell(fd);
    fseek(fd, 0, SEEK_SET);

    rewind(fd);

    uint8_t *data = (uint8_t *)malloc(file_size + 1);
    if (!data) {
        fprintf(stderr, "Memory allocation for %s failed\n", filename);
        return NULL;
    }
    size_t read_len = fread(data, 1, file_size, fd);
    if (read_len != (size_t)(file_size)) {
        fprintf(stderr, "fread failed\n");
        free(data);
        fclose(fd);
        return NULL;
    }
    fclose(fd);

    return data;
}
// end testing

uint8_t hex_char_to_int(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    } else if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    } else if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    } else {
        fprintf(stderr, "Invalid hexadecimal character: %c\n", c);
        exit(EXIT_FAILURE);
    }
}

// Function to convert a hex string to a byte vector
uint8_t* hex_string_to_bytes(const char* hex_str, size_t* out_len) {
    size_t len = strlen(hex_str);
    if (len % 2 != 0) {
        fprintf(stderr, "Invalid hex string length\n");
        exit(EXIT_FAILURE);
    }

    size_t byte_len = len / 2;
    uint8_t* bytes = (uint8_t*)malloc(byte_len);
    if (bytes == NULL) {
        perror("Failed to allocate memory");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < byte_len; ++i) {
        bytes[i] = (hex_char_to_int(hex_str[2 * i]) << 4) | hex_char_to_int(hex_str[2 * i + 1]);
    }

    if (out_len != NULL) {
        *out_len = byte_len;
    }

    return bytes;
}

int
read_tokenizer(char *filename)
{
    FILE *fd = fopen(filename, "rb");
    if (!fd) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return -1;
    }
    fseek(fd, 0, SEEK_END);
    long file_size = ftell(fd);
    fseek(fd, 0, SEEK_SET);
    fclose(fd);

    return file_size;
}

int
main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model1.onnx> <model2.onnx> ... <modelN.onnx> <input.pb>\n", argv[0]);
        return 1;
    }

    struct timeval t1_inf, t2_inf;
    double elapsed_time;

    //test tract decryption
    EncryptionParameters *params = (EncryptionParameters *)malloc(sizeof(EncryptionParameters));
    if (!params) {
        fprintf(stderr, "Memory allocation for params failed\n");
        return 1;
    }
    uint8_t *tag = NULL;
    uint8_t *key = write_to_buffer("aes/key.bin");
    uint8_t *iv = write_to_buffer("aes/iv.bin");
    uint8_t *aad = write_to_buffer("aes/add_data.bin");
    if (!key || !iv || !aad) {
        fprintf(stderr, "Error writing to buffer\n");
        free(params);
        return 1;
    }
    params->key = key;
    params->iv = iv;
    params->aad = aad;

    // print the params
    fprintf(stderr, "Key: ");
    for (int i = 0; i < KEY_BYTES; i++) {
        fprintf(stderr, "%02x", params->key[i]);
    }
    fprintf(stderr, "\nIV: ");
    for (int i = 0; i < IV_BYTES; i++) {
        fprintf(stderr, "%02x", params->iv[i]);
    }
    fprintf(stderr, "\nAAD: ");
    for (int i = 0; i < ADD_DATA_BYTES; i++) {
        fprintf(stderr, "%02x", params->aad[i]);
    }
    fprintf(stderr, "\n");

    if (strcmp(argv[2], "tokenizer.json") == 0) {
        const char *tag_message = "108588b3719ee19b8b4d37fb7250cdbe";
        tag = (uint8_t *)malloc(TAG_BYTES * 2);
        if (!tag) {
            fprintf(stderr, "Memory allocation for tag failed\n");
            return 1;
        }
        memcpy(tag, tag_message, TAG_BYTES * 2);
        //tag = write_to_buffer(tag_message);
        // if (!tag) {
        //     fprintf(stderr, "Error writing to buffer\n");
        //     return 1;
        // }
        params->tag = tag;
        fprintf(stderr, "\nTag: %s\n", params->tag);
        // for (int i = 0; i < TAG_BYTES; i++) {
        //     fprintf(stderr, "%02x", params->tag[i]); //%02x
        // }
        // fprintf(stderr, "\n");

        char *model_path= argv[1];
        EncryptionParameters *params_weights = NULL;
        uint8_t *tag_weights = NULL, *key_weights = NULL, *iv_weights = NULL, *aad_weights = NULL;
        if (strcmp(model_path, "albert") != 0 && strcmp(model_path, "cerebras-gpt") != 0) {
            params_weights = (EncryptionParameters *)malloc(sizeof(EncryptionParameters));
            if (!params_weights) {
                fprintf(stderr, "Memory allocation for params_weights failed\n");
                return 1;
            }
            
            key_weights = write_to_buffer("aes/key2.bin");
            iv_weights = write_to_buffer("aes/iv2.bin");
            aad_weights = write_to_buffer("aes/add_data2.bin");
            if (!key_weights || !iv_weights || !aad_weights) {
                fprintf(stderr, "Error writing to buffer for weights\n");
                free(params);
                return 1;
            }
            params_weights->key = key_weights;
            params_weights->iv = iv_weights;
            params_weights->aad = aad_weights;

            fprintf(stderr, "\nNow for weights: \nKey: ");
            for (int i = 0; i < KEY_BYTES; i++) {
                fprintf(stderr, "%02x", params_weights->key[i]);
            }
            fprintf(stderr, "\nIV: ");
            for (int i = 0; i < IV_BYTES; i++) {
                fprintf(stderr, "%02x", params_weights->iv[i]);
            }
            fprintf(stderr, "\nAAD: ");
            for (int i = 0; i < ADD_DATA_BYTES; i++) {
                fprintf(stderr, "%02x", params_weights->aad[i]);
            }
            fprintf(stderr, "\n");
        
            const char *tag_message_weights = "4ecc8e4742c0e69773699a2fb2ead084";
            
            tag_weights = (uint8_t *)malloc(TAG_BYTES * 2);
            if (!tag) {
                fprintf(stderr, "Memory allocation for tag failed\n");
                return 1;
            }
            memcpy(tag_weights, tag_message_weights, TAG_BYTES * 2);
            params_weights->tag = tag_weights;
            fprintf(stderr, "Tag weights: %s\n", params_weights->tag);
        }


        char *model_for_path = NULL, *model_for_path_weights = NULL;
        char *tokenizer_path;
        if (strcmp(model_path, "albert") == 0) {
            model_for_path = "/hdd/papafrkon/albert-base-v2/albert-base-v2.onnx";
                tokenizer_path = "/hdd/papafrkon/dAIEdgeServer/models/albert-large-v2/test_data_set_0/tokenizer.json";
        } else if (strcmp(model_path, "cerebras-gpt") == 0) {
            model_for_path = "../../../github_repo/InferONNX/models/cerebras-gpt-256M/cerebras-gpt-256M.onnx";
            tokenizer_path = "../../../github_repo/InferONNX/models/cerebras-gpt-256M/test_data_set_0/tokenizer.json";
        } else if (strcmp(model_path, "qwen") == 0) {
            model_for_path = "../../../github_repo/InferONNX/models/qwen2.5-0.5B/qwen2.5-0.5B.onnx";
            model_for_path_weights = "./model.onnx_data";
            tokenizer_path = "../../../github_repo/InferONNX/models/qwen2.5-0.5B/test_data_set_0/tokenizer.json";
        } else if (strcmp(model_path, "llama") == 0) {
            model_for_path = "../../../github_repo/InferONNX/models/llama3.2-1B/llama3.2-1B.onnx";
            model_for_path_weights = "../../../github_repo/InferONNX/models/llama3.2-1B/model.onnx_data";
            tokenizer_path = "../../../github_repo/InferONNX/models/llama3.2-1B/test_data_set_0/tokenizer.json";
        } else if (strcmp(model_path, "deepseek") == 0) {
            model_for_path = "../../../github_repo/InferONNX/models/deepseek-coder-1.3b-base/deepseek-coder-1.3b-base.onnx";
            model_for_path_weights = "../../../github_repo/InferONNX/models/deepseek-coder-1.3b-base/model.onnx_data";
            tokenizer_path = "../../../github_repo/InferONNX/models/deepseek-coder-1.3b-base/test_data_set_0/tokenizer.json";
        } else {
            fprintf(stderr, "Wrong NLP model!\n");
            return 1;
        }

        int nums = 1;
        for (int i = 0; i < nums; i++) {
            int tokenizer_size = read_tokenizer(tokenizer_path);
            uint8_t* tokenizer = write_to_buffer(tokenizer_path);
        
            char *inference = NULL;
            MyInferenceModel *inference_models = NULL;
            int is_albert = strstr(model_path, "albert") != NULL;

#if USE_MEMORY_ONLY == 1        
        tract_load_nlp_model(model_for_path, params, params_weights, &inference_models);
        assert(inference_models);
#endif
            gettimeofday(&t1_inf, NULL);
            if (is_albert) {
                    tract_run_albert(model_for_path, tokenizer, tokenizer_size, &inference, params, inference_models ? &inference_models : NULL);
            } else {
                tract_run_latest_models(model_for_path, tokenizer, tokenizer_size, &inference, params, params_weights, 5, "Hi", inference_models ? &inference_models : NULL);
            }
            gettimeofday(&t2_inf, NULL);
            elapsed_time = (t2_inf.tv_sec - t1_inf.tv_sec) * 1000.0;      // sec to ms
            elapsed_time += (t2_inf.tv_usec - t1_inf.tv_usec) / 1000.0;   // us to ms

            if (inference) {
                fprintf(stderr, "%s\nInference time to run a model: %f ms\n", inference, elapsed_time);
                tract_free_cstring(inference);
            }
            free(tokenizer);
        }

        free(tag);
        free(key);
        free(iv);
        free(aad);
        free(params);
        if (params_weights) {
            free(tag_weights);
            free(key_weights);
            free(iv_weights);
            free(aad_weights);
            free(params_weights);
        }
	    tract_free_onig();
        return 0;
    }

    for (int i = 1; i < argc-1; i++) {
        FILE *fd = fopen(argv[i], "rb");
        if (!fd) {
            fprintf(stderr, "Error opening model file %s\n", argv[i]);
            return 1;
        }
        fclose(fd);
    }

    FILE *fd = fopen(argv[argc - 1], "rb");
    if (!fd) {
        fprintf(stderr, "Error opening model_input file");
        return 1;
    }
    
    size_t *shape = decode_pb(fd);
    int calculated_shape = shape[0] * shape[1] * shape[2] * shape[3];
    fprintf(stderr, "Input shape: %zu %zu %zu %zu\n", shape[0], shape[1], shape[2], shape[3]);

    float *image = (float *) malloc( calculated_shape * sizeof(float));
    int image_floats = fread(image, sizeof(float), calculated_shape, fd);
    fprintf(stderr, "Read %d floats\n", image_floats);
    assert(image_floats == calculated_shape);
    fclose(fd);

    prediction** preds = init_predictions(argc-1);
    if (!preds) {
        return 1;
    }

    check(tract_value_from_bytes(TRACT_DATUM_TYPE_F32, 4, shape, image, &preds[0]->output));
    free(image);

    for (int i = 1; i < argc-1; i++) {
        int i_size=0, k=i;
        while (k != 0) {
            k /= 10;
            i_size++;
        }
        char tag_message[] = "835705756bf8ac03cf25b8b4dd2fcbfc";
        tag = (uint8_t *)malloc(TAG_BYTES * 2);
        if (!tag) {
            fprintf(stderr, "Memory allocation for tag failed\n");
            free_predictions(preds, argc-1);
            return 1;
        }
        memcpy(tag, tag_message, TAG_BYTES * 2);
        if (!tag) {
            fprintf(stderr, "Error writing to buffer\n");
            free_predictions(preds, argc-1);
            return 1;
        }
        params->tag = tag;
        fprintf(stderr, "\nTag: ");
        for (int i = 0; i < TAG_BYTES; i++) {
            fprintf(stderr, "%02x", params->tag[i]);
        }
        fprintf(stderr, "\n");
        preds[i] = inference(argv[i], preds[i-1]->output, NULL, preds[i], params);
        if (!preds[i]) {
            fprintf(stderr, "Error running inference for model %s\n", argv[i]);
            free_predictions(preds, argc-1);
            free(tag);
            free(key);
            free(iv);
            free(aad);
            free(params);
            return 1;
        }
        free(tag);
    }

    free_predictions(preds, argc-1);
    fprintf(stderr, "All done\n");

    // testing
    free(key);
    free(iv);
    free(aad);
    free(params);

    return 0;
}
