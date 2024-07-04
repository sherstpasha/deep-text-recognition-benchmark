from recog import load_model, recognize_text

config_path = r"config.yaml"
model_path = r"saved_models\TPS-ResNet-BiLSTM-Attn\best_accuracy.pth"
image_path = r"C:\Users\user\Desktop\megacorpus\stackmix_reports_large\img0"

model, converter, opt = load_model(config_path, model_path)
for i in range(200):

    recognized_text = recognize_text(model, converter, opt, image_path + f'\image_0000{i + 1}.png')
    print('Predicted text:', recognized_text)
    print('Predicted text:', recognized_text)