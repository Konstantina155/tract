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
    struct timeval t1, t2;
    double elapsedTime;

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
    check(tract_inference_model_into_optimized(&inference_model,&model));
    assert(model);

    // Make the model runnable
    TractRunnable *runnable = NULL;
    check(tract_model_into_runnable(&model, &runnable));
    assert(runnable);
    assert(!model);

    TractValue* output = NULL;

    gettimeofday(&t1, NULL);

    // simple stateless run...
    TractValue *inputs[] = { input, input2 };
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

    
    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
    printf("Model run in %f ms.\n", elapsedTime);

    inf->output = output;
    inf->pred = max;
    inf->category = argmax;

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

    rewind(fd);

    return file_size;
}

int
main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model1.onnx> <model2.onnx> ... <modelN.onnx> <input.pb>\n", argv[0]);
        return 1;
    }

    //test tract decryption
    EncryptionParameters *params = (EncryptionParameters *)malloc(sizeof(EncryptionParameters));
    if (!params) {
        fprintf(stderr, "Memory allocation for params failed\n");
        return 1;
    }
    size_t len;
    uint8_t *key = hex_string_to_bytes("65ddc559144ae2aecfe4b10432cb8a53a8e62a20957e902005b07e0509352d02", &len);
    uint8_t *iv = hex_string_to_bytes("a0792200b9c64095886a94d7", &len);
    uint8_t *aad = hex_string_to_bytes("f72ea3659d262b1d03b14a0a53a3c988cfadb418cf77aaeaee5544755f694484e7f2c787833f91a1c6e2c710ecdda85349fa49396009ad8b10e54517f1ab95f0", &len);
    uint8_t *tag = NULL;
    // key = write_to_buffer("aes/key.bin");
    // iv = write_to_buffer("aes/iv.bin");
    // aad = write_to_buffer("aes/add_data.bin");
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
    // end testing

    if (strcmp(argv[1], "albert") == 0) {
        char tag_message[] = "92e2e37cc8b8e7ef9ef7e2f74b5984fa";
        tag = (uint8_t *)malloc(TAG_BYTES * 2);
        if (!tag) {
            fprintf(stderr, "Memory allocation for tag failed\n");
            return 1;
        }
        memcpy(tag, tag_message, TAG_BYTES * 2);
        //tag = write_to_buffer(tag_message);
        if (!tag) {
            fprintf(stderr, "Error writing to buffer\n");
            return 1;
        }
        params->tag = tag;
        fprintf(stderr, "\nTag: ");
        for (int i = 0; i < TAG_BYTES; i++) {
            fprintf(stderr, "%02x", params->tag[i]);
        }
        fprintf(stderr, "\n");

        char *model_for_path = "../../examples/pytorch-albert-v2/albert/encrypted_model.onnx";
        char* inference = NULL;
        int tokenizer_size = read_tokenizer("../../examples/pytorch-albert-v2/albert/tokenizer.json");
        const uint8_t* tokenizer = write_to_buffer("../../examples/pytorch-albert-v2/albert/tokenizer.json");
        check(tract_run_albert(model_for_path, tokenizer, tokenizer_size, &inference, params));
        fprintf(stderr, "Inference: %s\n", inference);

        free(inference);
        free(tag);
        free(key);
        free(iv);
        free(aad);
        free(params);
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

    //Hint for splitting the models into a node that is part of cut from parent node (circle)
    //The inference of the last model is gonna take the output of the 2 previous models, like input2, input3
    for (int i = 1; i < argc-1; i++) {
        int i_size=0, k=i;
        while (k != 0) {
            k /= 10;
            i_size++;
        }
        char tag_message[] = "dcec09760a5fed9c54a093554631f5df";
        tag = (uint8_t *)malloc(TAG_BYTES * 2);
        if (!tag) {
            fprintf(stderr, "Memory allocation for tag failed\n");
            free_predictions(preds, argc-1);
            return 1;
        }
        memcpy(tag, tag_message, TAG_BYTES * 2);
        //tag = write_to_buffer(tag_message);
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