#include "rotate.h"

static float in_arr[1 * 3 * 1024 * 1024];
static float out0_arr[1 * 21504 * 7];

void Rotatation::loadEngine(const std::string& path) {
    size_t size{ 0 };
    char* trtModelStream{ nullptr };
    std::ifstream file(path, std::ios::binary);

    if (file.good()) {
        file.seekg(0, std::ios::end);
        size = file.tellg();
        file.seekg(0, std::ios::beg);
        trtModelStream = new char[size];
        file.read(trtModelStream, size);
        file.close();
    }

    runtime_ = createInferRuntime(logger_);
    engine_ = runtime_->deserializeCudaEngine(trtModelStream, size);
    context_ = engine_->createExecutionContext();

    delete[] trtModelStream;
}


cv::Mat Rotatation::preprocessImg(cv::Mat& img, const int& input_w, const int& input_h, int& padw, int& padh) {
    int w, h, x, y;
    float r_w = input_w / (img.cols * 1.0);
    float r_h = input_h / (img.rows * 1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    }
    else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(127, 127, 127));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    padw = (input_w - w) / 2;
    padh = (input_h - h) / 2;
    return out;
}



void  Rotatation::LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
	bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color)
{
	if (false) {
		int maxLen = MAX(image.rows, image.cols);
		outImage = cv::Mat::zeros(cv::Size(maxLen, maxLen), CV_8UC3);
		image.copyTo(outImage(cv::Rect(0, 0, image.cols, image.rows)));
		params[0] = 1;
		params[1] = 1;
		params[3] = 0;
		params[2] = 0;
	}

	cv::Size shape = image.size();
	float r = std::min((float)newShape.height / (float)shape.height,
		(float)newShape.width / (float)shape.width);
	if (!scaleUp)
		r = std::min(r, 1.0f);

	float ratio[2]{ r, r };
	int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

	auto dw = (float)(newShape.width - new_un_pad[0]);
	auto dh = (float)(newShape.height - new_un_pad[1]);

	if (autoShape)
	{
		dw = (float)((int)dw % stride);
		dh = (float)((int)dh % stride);
	}
	else if (scaleFill)
	{
		dw = 0.0f;
		dh = 0.0f;
		new_un_pad[0] = newShape.width;
		new_un_pad[1] = newShape.height;
		ratio[0] = (float)newShape.width / (float)shape.width;
		ratio[1] = (float)newShape.height / (float)shape.height;
	}

	dw /= 2.0f;
	dh /= 2.0f;

	if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
	{
		cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
	}
	else {
		outImage = image.clone();
	}

	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));
	params[0] = ratio[0];
	params[1] = ratio[1];
	params[2] = left;
	params[3] = top;
	cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}


void Rotatation::totalInference(const std::string& directory){

    auto dji_names = listJpgFiles(directory);

    for(auto dji_name : dji_names){

        std::cout<<dji_name<<std::endl;

        cv::Mat image = cv::imread(dji_name);

        int imageWidth = image.cols;
        int imageHeight = image.rows;
        int k = 1024;
        int stride = 824;

        std::vector<cv::RotatedRect> total_bboxes;
        std::vector<float> total_scores;
        std::vector<int>total_indices;
        std::vector<int>total_label_idxs;
        int cnt = 0;

        for (int x = 0; x < imageWidth; x += stride) {
            for (int y = 0; y < imageHeight; y += stride) {
  
                int currentK = std::min(k, imageWidth - x);
                int currentStride = std::min(k, imageHeight - y);

                cv::Rect roi(x, y, currentK, currentStride);
                cv::Mat croppedImage(image, roi);
             
                std::string temp_path = "../cropped/temp_"+std::to_string(cnt)+".png";
                cv::imwrite(temp_path,croppedImage);
                cnt ++;
         
                cv::Mat img_roi = cv::imread(temp_path);

                int32_t input_index = engine_->getBindingIndex(images_);
                int32_t output0_index = engine_->getBindingIndex(output0_);

                void* buffers[2];
                cudaMalloc(&buffers[input_index], BATCH_SIZE * CHANNELS * INPUT_W * INPUT_H * sizeof(float));
                cudaMalloc(&buffers[output0_index], BATCH_SIZE * OUTPUT0_BOXES * OUTPUT0_ELEMENT * sizeof(float));

                cv::Mat pr_img;
                cv::Vec4d params;
                LetterBox(img_roi, pr_img, params, cv::Size(INPUT_W, INPUT_H),false,false,true,32,cv::Scalar(114, 114, 114));

                for (int i = 0; i < INPUT_W * INPUT_H; i++) {
                    in_arr[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
                    in_arr[i + INPUT_W * INPUT_H] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
                    in_arr[i + 2 * INPUT_W * INPUT_H] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
                }

                cudaStream_t stream;
                cudaStreamCreate(&stream);
                cudaMemcpyAsync(buffers[input_index], in_arr, BATCH_SIZE * CHANNELS * INPUT_W * INPUT_H * sizeof(float), cudaMemcpyHostToDevice, stream);
                context_->enqueueV2(buffers, stream, nullptr);
                cudaMemcpyAsync(out0_arr, buffers[output0_index], BATCH_SIZE * OUTPUT0_BOXES * OUTPUT0_ELEMENT * sizeof(float), cudaMemcpyDeviceToHost, stream);

                cudaStreamSynchronize(stream);
                cudaStreamDestroy(stream);
                cudaFree(buffers[input_index]);
                cudaFree(buffers[output0_index]);

                float r_w = INPUT_W / (img_roi.cols * 1.0);
                float r_h = INPUT_H / (img_roi.rows * 1.0);


                int net_width = OUTPUT0_ELEMENT;
                float* pdata = out0_arr;

                std::vector<cv::RotatedRect> bboxes;
                std::vector<float> scores;
                std::vector<int>indices;
                std::vector<int>label_idxs;

                for (int i = 0; i < OUTPUT0_BOXES; i++) {

                    float* score_ptr = std::max_element(pdata + 4, pdata + 4 + CLASSES);
                    float box_score = *score_ptr;
                    int label_index = score_ptr - (pdata + 4);

                    if (box_score >= CONF_THRESHOLD) {

                        float x_org = (pdata[0] - params[2]) / params[0] + x;
                        float y_org = (pdata[1] - params[3]) / params[1] + y;
                        float w = pdata[2] / params[0];
                        float h = pdata[3] / params[1];
                        float angle = pdata[4+CLASSES] /CV_PI *180.0;

                        cv::RotatedRect rotate_rect = cv::RotatedRect(cv::Point2f(x_org,y_org),cv::Size2f(w,h),angle);
                        bboxes.push_back(rotate_rect);
                        scores.push_back(box_score);
                        label_idxs.push_back(label_index);

                    }

                    pdata += net_width; 
                }

                cv::dnn::NMSBoxes(bboxes,scores,CONF_THRESHOLD,NMS_THRESHOLD,indices);
                
                for(auto idx : indices){

                    int label_index= label_idxs[idx];
                    std::string label = class_names_rot[label_index];
                    cv::RotatedRect rec = bboxes[idx];
                    float score = scores[idx];

                    total_bboxes.push_back(rec);
                    total_scores.push_back(score);
                    total_label_idxs.push_back(label_index);
                    
                    // cv::Point2f ps[4] ={};
                    // rec.points(ps);
                    // cv::line(img_roi,ps[0],ps[1],cv::Scalar(0,255,0),2);
                    // cv::line(img_roi,ps[1],ps[2],cv::Scalar(0,255,0),2);
                    // cv::line(img_roi,ps[2],ps[3],cv::Scalar(0,255,0),2);
                    // cv::line(img_roi,ps[3],ps[0],cv::Scalar(0,255,0),2);
                 
                }
                // cv::imwrite("../cropped/sss_test.png",img_roi);

            }


        }

        cv::dnn::NMSBoxes(total_bboxes,total_scores,CONF_THRESHOLD,NMS_THRESHOLD,total_indices);
        std::cout<<"single big image over!"<<std::endl;

//save json file
        int last_slash_pos = dji_name.find_last_of("/\\");
        int last_dot_pos = dji_name.find_last_of(".");
        std::string img_name = dji_name.substr(last_slash_pos + 1, last_dot_pos - last_slash_pos - 1);
     
        json j;
        j["version"] = "0.3.3";
        j["flags"]={};
        j["shapes"]={};
        j["imagePath"]= img_name + ".png";
        j["imageData"] ={};
        j["imageHeight"] =imageHeight ;
        j["imageWidth"] = imageWidth;

        for(auto idx: total_indices){
            int label_index= total_label_idxs[idx];
            std::string label = class_names_rot[label_index];
            cv::RotatedRect rec = total_bboxes[idx];

            cv::Point2f ps[4] ={};

            rec.points(ps);
            
            json j_temp;
            j_temp["label"] = label;
            j_temp["text"]="";
            j_temp["group_id"]={};
            j_temp["shape_type"]="polygon";
            j_temp["flags"] = {};
            j_temp["points"] ={}; 
            for(auto &p: ps){
                j_temp["points"].push_back({p.x,p.y});
            }
            j["shapes"].push_back(j_temp);

        }
        
        if(!j["shapes"].empty()){

            std::string  out_json = "/home/ubuntu/wjd/rotate/images/"+ img_name +  + ".json";
            std::ofstream o(out_json);
            o << std::setw(4) << j << std::endl;

        }
    }

}


void Rotatation::rotateInference(cv::Mat & img, json &j){

    int32_t input_index = engine_->getBindingIndex(images_);
    int32_t output0_index = engine_->getBindingIndex(output0_);

    void* buffers[2];
    cudaMalloc(&buffers[input_index], BATCH_SIZE * CHANNELS * INPUT_W * INPUT_H * sizeof(float));
    cudaMalloc(&buffers[output0_index], BATCH_SIZE * OUTPUT0_BOXES * OUTPUT0_ELEMENT * sizeof(float));

    cv::Mat pr_img;
    cv::Vec4d params;
    LetterBox(img, pr_img, params, cv::Size(INPUT_W, INPUT_H),false,false,true,32,cv::Scalar(114, 114, 114));

    for (int i = 0; i < INPUT_W * INPUT_H; i++) {
        in_arr[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
        in_arr[i + INPUT_W * INPUT_H] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
        in_arr[i + 2 * INPUT_W * INPUT_H] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(buffers[input_index], in_arr, BATCH_SIZE * CHANNELS * INPUT_W * INPUT_H * sizeof(float), cudaMemcpyHostToDevice, stream);
    context_->enqueueV2(buffers, stream, nullptr);
    cudaMemcpyAsync(out0_arr, buffers[output0_index], BATCH_SIZE * OUTPUT0_BOXES * OUTPUT0_ELEMENT * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(buffers[input_index]);
    cudaFree(buffers[output0_index]);

    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);

    int net_width = OUTPUT0_ELEMENT;
    float* pdata = out0_arr;

    std::vector<cv::RotatedRect> bboxes;
    std::vector<float> scores;
    std::vector<int>indices;
    std::vector<int>label_idxs;

    for (int i = 0; i < OUTPUT0_BOXES; i++) {

        float* score_ptr = std::max_element(pdata + 4, pdata + 4 + CLASSES);
        float box_score = *score_ptr;
        int label_index = score_ptr - (pdata + 4);

        if (box_score >= CONF_THRESHOLD) {

            float x_org = (pdata[0] - params[2]) / params[0] ;
            float y_org = (pdata[1] - params[3]) / params[1] ;
            float w = pdata[2] / params[0];
            float h = pdata[3] / params[1];
            float angle = pdata[4+CLASSES] /CV_PI *180.0;

            cv::RotatedRect rotate_rect = cv::RotatedRect(cv::Point2f(x_org,y_org),cv::Size2f(w,h), angle);
            bboxes.push_back(rotate_rect);
            scores.push_back(box_score);
            label_idxs.push_back(label_index);

        }

        pdata += net_width; 
    }

    cv::dnn::NMSBoxes(bboxes,scores,CONF_THRESHOLD,NMS_THRESHOLD,indices);
                
    for(auto idx : indices){

        int label_index= label_idxs[idx];
        std::string label = class_names_rot[label_index];
        cv::RotatedRect rec = bboxes[idx];
        float score = scores[idx];

        cv::Point2f ps[4] ={};
        rec.points(ps);

        json j_temp;
        j_temp["label"] = label;
        j_temp["text"]="";
        j_temp["group_id"]={};
        j_temp["shape_type"]="polygon";
        j_temp["flags"] = {};
        j_temp["points"] ={}; 
        for(auto & p : ps){
            j_temp["points"].push_back({p.x, p.y});

        }
        
        j["shapes"].push_back(j_temp);

    }


}



void Rotatation::initDetection(){

    yolo_ = std::move(yolo::load(detection_path, yolo::Type::V8Seg));
    if (yolo_ == nullptr) return;

}


yolo::Image Rotatation::cvimg(const cv::Mat &image) { 

    return yolo::Image(image.data, image.cols, image.rows); 
}



std::vector<std::string> Rotatation::listJpgFiles(const std::string& directory) {
    DIR *dir;
    struct dirent *entry;
    std::vector<std::string> total_names;
    if ((dir = opendir(directory.c_str())) != NULL) {
        while ((entry = readdir(dir)) != NULL) {
            std::string filename = entry->d_name;
            // 检查文件名是否以.jpg结尾（忽略大小写）
            if (filename.length() >= 4 && strcasecmp(filename.substr(filename.length() - 4).c_str(), ".JPG") == 0) {
                
                auto org_img_name = directory + "/" + filename;
                total_names.push_back(org_img_name);

            }
        }
        closedir(dir);
    } 

    return total_names;
}




void Rotatation::single_inference() {

  auto total_names = listJpgFiles(dji_img_path);

  auto length = total_names.size();
  int number = 0;

  for(auto img_path : total_names){

        cv::Mat image = cv::imread(img_path);
        int imageWidth = image.cols;
        int imageHeight = image.rows;
        int k = 640;
        int stride = 400;

        std::vector<cv::Rect> bboxes;
        std::vector<float> scores;
        std::vector<int>indices;
        std::vector<int>label_indexs;
    
        int newcnt=0;
        for (int x = 0; x < imageWidth; x += stride) {
            for (int y = 0; y < imageHeight; y += stride) {
   
                int currentK = std::min(k, imageWidth - x);
                int currentStride = std::min(k, imageHeight - y);

                cv::Rect roi(x, y, currentK, currentStride);
                cv::Mat croppedImage(image, roi); 
      
                std::string out_path = "../tempimg/"+std::to_string(newcnt)+".jpg";
                cv::imwrite(out_path,croppedImage);

                newcnt++;
          
                cv::Mat small_img = cv::imread(out_path);
                
                auto objs = yolo_->forward(cvimg(small_img));

            
   
                for(auto & obj : objs){      
                    if(obj.class_label == 4) continue;//rail_big TODO
                        obj.left += x;
                        obj.top +=y;
                        obj.right+=x;
                        obj.bottom +=y;
                        
                        cv::Rect2i org_rect = cv::Rect(cv::Point2i(obj.left,obj.top),cv::Point2i(obj.right,obj.bottom));
                        float score = obj.confidence;
                        int label = obj.class_label;

                        bboxes.push_back(org_rect);
                        scores.push_back(score);
                        label_indexs.push_back(label);
        
                }
            }
        }

        cv::dnn::NMSBoxes(bboxes,scores,CONF_THRESHOLD,NMS_THRESHOLD,indices);

        
        int last_slash_pos = img_path.find_last_of("/\\");
        int last_dot_pos = img_path.find_last_of(".");
        std::string img_name = img_path.substr(last_slash_pos + 1, last_dot_pos - last_slash_pos - 1);

        json j;
        j["version"] = "0.3.3";
        j["flags"]={};
        j["shapes"]={};
        j["imagePath"]= img_name + ".JPG";
        j["imageData"] ={};
        j["imageHeight"] =imageHeight ;
        j["imageWidth"] = imageWidth;


        for(auto idx: indices){
            int label_index= label_indexs[idx];
            std::string label = class_names_det[label_index];
            cv::Rect rec = bboxes[idx];
            auto a = rec.tl();
            auto b = rec.br(); 

            json j_temp;
            j_temp["label"] = label;
            j_temp["text"]="";
            j_temp["points"] ={{a.x,a.y},{b.x,b.y}}; 
            j_temp["group_id"]={};
            j_temp["shape_type"]="rectangle";
            j_temp["flags"] = {};
            j["shapes"].push_back(j_temp);
        }
        

        rotateInference(image, j);


        if(!j["shapes"].empty()){

            std::string out_json = dji_img_path +"/" + img_name +  + ".json";
            // std::string  out_json = "/home/ubuntu/wjd/dajiang/"+ img_name +  + ".json";
            std::ofstream o(out_json);
            o << std::setw(4) << j << std::endl;

            auto ratio = 1.0* number / length;
            std::cout<<"already complete "<<img_name<< " "<< number<< "/ "<< length<< " = "<<ratio<<std::endl;
            number++;

        }

        //   if (obj.seg) {
        //         cv::Mat mask =  cv::Mat(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data);
        //         cv::resize(mask,mask,cv::Size(obj.right-obj.left,obj.bottom-obj.top),0,0,cv::INTER_CUBIC);
        //    for(int j = 0 ; j < mask.rows ; j++){
        //       for(int k = 0 ; k<mask.cols; k++){
        //           if(mask.at<uchar>(j,k) >=120  ){
        //             mask.at<uchar>(j,k) =255;
        //           }else{
        //             mask.at<uchar>(j,k) = 0;
        //           }
        //       }
        //     }

    }
   
}

void Rotatation::run(){

    this->initDetection();
    this->loadEngine(obb_path_);
    this->single_inference();


}