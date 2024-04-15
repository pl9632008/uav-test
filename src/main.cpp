
#include <nlohmann/json.hpp>
#include <fstream>
using json = nlohmann::json;
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/opencv.hpp>
#include <dirent.h>

#include "cpm.hpp"
#include "infer.hpp"
#include "yolo.hpp"



const char* class_names[]={"fastener_normal","fastener_abnormal","fastener_stone","fastener_missing",
                    "sleeper_normal","sleeper_abnormal" ,"rail_big"
};

std::vector<std::string> class_names2{"fastener_normal","fastener_abnormal","fastener_stone","fastener_missing",
                    "sleeper_normal","sleeper_abnormal" ,"rail_big"};



yolo::Image cvimg(const cv::Mat &image) { return yolo::Image(image.data, image.cols, image.rows); }



std::vector<std::string> listJpgFiles(const std::string& directory) {
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


void listJpgFiles2(const std::string& directory,std::vector<std::string>& json_names,std::vector<std::string> &dji_names) {
    DIR *dir;
    struct dirent *entry;

    if ((dir = opendir(directory.c_str())) != NULL) {
        while ((entry = readdir(dir)) != NULL) {
            std::string filename = entry->d_name;
            // 检查文件名是否以.jpg结尾（忽略大小写）
            if (filename.length() >= 5 && strcasecmp(filename.substr(filename.length() - 5).c_str(), ".json") == 0) {
                
                auto json_name = directory + "/" + filename;

                int last_slash_pos = json_name.find_last_of("/\\");
                int last_dot_pos = json_name.find_last_of(".");
                std::string img_name = json_name.substr(last_slash_pos + 1, last_dot_pos - last_slash_pos - 1);

                auto dji_name = directory + "/" + img_name +".JPG";
                  
                json_names.push_back(json_name);
                dji_names.push_back(dji_name);

            }
        }
        closedir(dir);
    } 

}


void listJpgFiles3(const std::string& directory,std::vector<std::string>& json_names) {
    DIR *dir;
    struct dirent *entry;

    if ((dir = opendir(directory.c_str())) != NULL) {
        while ((entry = readdir(dir)) != NULL) {
            std::string filename = entry->d_name;
            // 检查文件名是否以.jpg结尾（忽略大小写）
            if (filename.length() >= 5 && strcasecmp(filename.substr(filename.length() - 5).c_str(), ".json") == 0) {
                
                auto json_name = directory + "/" + filename;
                  
                json_names.push_back(json_name);


            }
        }
        closedir(dir);
    } 

    
}




int cnt = 0;
void single_test() {

  auto yolo = yolo::load("../model/fastener.engine", yolo::Type::V8Seg);
  if (yolo == nullptr) return;

  std::string dji_img_path = "/home/ubuntu/wjd/uav/out";

    auto total_names = listJpgFiles(dji_img_path);

    auto start1 = std::chrono::system_clock::now();
    
    for(auto i : total_names){

        cv::Mat img = cv::imread(i);

        std::cout<<i<<std::endl;
        auto objs = yolo->forward(cvimg(img));

        bool flag = false;
        for(auto obj : objs){

            flag = true;
            cv::rectangle(img, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                  cv::Scalar(255, 0, 0), 5);

                auto name = class_names[obj.class_label];

                 auto caption = cv::format("%s %.2f", name, obj.confidence);
                int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                cv::rectangle(img, cv::Point(obj.left - 3, obj.top - 33),
                            cv::Point(obj.left + width, obj.top), cv::Scalar(0, 255, 0), -1);
                cv::putText(img, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);



        }
        if(flag){

            std::string name_path = "../out2/"+std::to_string(cnt)+"_.jpg";
            cv::imwrite(name_path,img);
            cnt++;  

        }

    }

}



void test4(cv::Mat & H , cv::Mat & img1 , cv::Mat & img2){

    std::ifstream f("../DJI_20240227110246_0003.json");
    json j = json::parse(f);
    std::ofstream out("../config.ini");
    int index = 0;
    for(auto & i : j["shapes"]){

        std::string label = i["label"];
        out<<"["<<label<<index<<"]\n";


        if(i["shape_type"] == "rectangle"){

            cv::Point2f p1(i["points"][0][0],i["points"][0][1]);      
            cv::Point2f p2;
            cv::Point2f p3(i["points"][1][0],i["points"][1][1]);
            cv::Point2f p4;

            auto w = p3.x-p1.x;
            auto h = p3.y-p1.y;

            p2.x = p1.x+w;
            p2.y = p1.y;
            p4.x = p1.x;
            p4.y = p1.y+h;

            std::vector<cv::Point2f> img_1_corners{p1,p2,p3,p4};

            for(auto &s : img_1_corners){

                  s.x *=0.1;
                  s.y *=0.1;

              std::cout<<s<<std::endl;
            }

            

            std::vector<cv::Point2f> img_2_corners(4);
            perspectiveTransform( img_1_corners, img_2_corners, H);

            std::cout<<"-------------"<<std::endl;

              for(auto &s : img_2_corners){

                        s.x *=10;
                        s.y *=10;
                        std::cout<<s<<std::endl;
                        
                  }
              auto rotated_rect = cv::RotatedRect(img_2_corners[0],img_2_corners[1],img_2_corners[2]);

              cv::Size si = rotated_rect.size;   


            for(int idx = 0 ; idx < 4 ; idx++){
              
              if(label == "fastener_normal"){

                    cv::line(img2,img_2_corners[idx%4],img_2_corners[(idx+1)%4],cv::Scalar(255,0,0),4);
              }
              if(label=="fastener_abnormal"){


                cv::line(img2,img_2_corners[idx%4],img_2_corners[(idx+1)%4],cv::Scalar(140,230,240),4);

              }
              
            }

            index++;  
            // cv::putText(img2,label,p1,2,2,cv::Scalar(0,0,0));
            out<<"point1 = "<<img_2_corners[0].x<<","<<img_2_corners[0].y<<"\n";
            out<<"point2 = "<<img_2_corners[2].x<<","<<img_2_corners[2].y<<"\n";

        }else if(i["shape_type"]=="polygon"){
            
            std::vector<cv::Point2f> img_1_corners;
            for(auto & item: i["points"]){
                cv::Point2f p(item[0],item[1]);

                img_1_corners.push_back(p);

            }

            std::vector<cv::Point2f> img_2_corners(img_1_corners.size());
            perspectiveTransform( img_1_corners, img_2_corners, H);
            int len = img_1_corners.size();
            for(int idx = 0 ; idx < len ; idx++){
              cv::line(img2,img_2_corners[idx%len],img_2_corners[(idx+1)%len],cv::Scalar(255,0,0));
            }

            // cv::putText(img2,label,img_2_corners[0],2,2,cv::Scalar(0,0,0));

        }

    }
    cv::imwrite("../img2_out.jpg",img2);


}




void test3(cv::Mat & img1, cv::Mat & img2){

    cv::Mat img_object = img1.clone();
    cv::Mat img_scene = img2.clone();

    cv::resize(img_object,img_object,cv::Size(),0.1,0.1);
   cv::resize(img_scene,img_scene,cv::Size(),0.1,0.1);

    cv::cvtColor(img_object,img_object,cv::COLOR_BGR2GRAY);
    cv::cvtColor(img_scene,img_scene,cv::COLOR_BGR2GRAY);

    int minHessian = 400;
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );
    std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;
    cv::Mat descriptors_object, descriptors_scene;
    detector->detectAndCompute( img_object, cv::noArray(), keypoints_object, descriptors_object );
    detector->detectAndCompute( img_scene, cv::noArray(), keypoints_scene, descriptors_scene );
    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    matcher->knnMatch( descriptors_object, descriptors_scene, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.75f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }


    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
    cv::Mat H = findHomography( obj, scene, cv::RANSAC );

    test4(H,img1,img2);


}

void test5(std::string & path,std::string  &path_json){

    cv::Mat image = cv::imread(path);

    std::ifstream json_file(path_json); 
    json j = json::parse(json_file);


    int last_slash_pos = path.find_last_of("/\\");
    int last_dot_pos = path.find_last_of(".");
    std::string img_name = path.substr(last_slash_pos + 1, last_dot_pos - last_slash_pos - 1);

    int imageWidth = image.cols;
    int imageHeight = image.rows;
    int k = 900;
    int stride = 600;


    // 分割图像并保存每个图块
    for (int x = 0; x < imageWidth; x += stride) {
        for (int y = 0; y < imageHeight; y += stride) {
            // 计算当前图块的实际大小
            int currentK = std::min(k, imageWidth - x);
            int currentStride = std::min(k, imageHeight - y);

            // 提取当前图块
            cv::Rect roi(x, y, currentK, currentStride);
            cv::Mat croppedImage(image, roi);

            std::string filename ="../croped_dingzi/"+ img_name + "_"+std::to_string(x) + "_" + std::to_string(y) + ".jpg";
            

            json j2;
            j2["version"] = "0.3.3";
            j2["flags"]={};
            j2["shapes"]={};
            j2["imagePath"]= img_name + "_"+std::to_string(x) + "_" + std::to_string(y) + ".jpg";
            j2["imageData"] ={};
            j2["imageHeight"] =currentStride ;
            j2["imageWidth"] = currentK;


            bool flag = false;
            for(auto & i : j["shapes"]){
                // if(i["shape_type"] == "rectangle"){
                if(i["label"] != "fastener_normal" && i["label"] != "fastener_abnormal" && i["label"] != "fastener_stone" && i["label"] != "fastener_missing" )
                continue;

                //   cv::Point2f p1(i["points"][0][0],i["points"][0][1]);      
                //   cv::Point2f p3(i["points"][1][0],i["points"][1][1]);

                //   auto a1 = p1.x - x;
                //   auto b1 = p1.y -y;

                //   auto a3 = p3.x - x;
                //   auto b3 = p3.y -y;

                //   if(  ( 0<a1 && a1<currentK &&  0<b1 && b1<currentStride) &&
                //        ( 0<a3 && a3<currentK &&  0<b3 && b3<currentStride) ){
                        
                //           flag = true;
                //           json j2_temp;
                //           j2_temp["label"] = i["label"];
                //           j2_temp["text"]="";
                //           j2_temp["points"] ={{a1,b1},{a3,b3}}; 
                //           j2_temp["group_id"]={};
                //           j2_temp["shape_type"]=i["shape_type"];
                //           j2_temp["flags"] = {};
                      
                          
                //           j2["shapes"].push_back(j2_temp);
                //        }

                // }

                cv::Point2f p1(i["points"][0][0],i["points"][0][1]);                      
                cv::Point2f p2(i["points"][1][0],i["points"][1][1]);  
                cv::Point2f p3(i["points"][2][0],i["points"][2][1]);
                cv::Point2f p4(i["points"][3][0],i["points"][3][1]);


                  auto a1 = p1.x - x;
                  auto b1 = p1.y -y;

                  auto a2 = p2.x-x;
                  auto b2 = p2.y-y;
                  
                  auto a4 = p4.x-x;
                  auto b4 = p4.y-y;

                  auto a3 = p3.x - x;
                  auto b3 = p3.y -y;

                  if(  ( 0<a1 && a1<currentK &&  0<b1 && b1<currentStride) &&
                       ( 0<a3 && a3<currentK &&  0<b3 && b3<currentStride) &&
                       ( 0<a2 && a2<currentK &&  0<b2 && b2<currentStride) &&
                       ( 0<a4 && a4<currentK &&  0<b4 && b4<currentStride) 
                        ){
                        
                          flag = true;
                          json j2_temp;
                          j2_temp["label"] = i["label"];
                          j2_temp["text"]="";
                          j2_temp["points"] ={{a1,b1},{a2,b2},{a3,b3},{a4,b4}}; 
                          j2_temp["group_id"]={};
                          j2_temp["shape_type"]=i["shape_type"];
                          j2_temp["flags"] = {};
                                     
                          j2["shapes"].push_back(j2_temp);
                   }

            }
            if(flag){

                std::string  out_json = "../croped_dingzi/"+ img_name + "_"+std::to_string(x) + "_" + std::to_string(y) + ".json";
                std::ofstream o(out_json);
                o << std::setw(4) << j2 << std::endl;
                cv::imwrite(filename, croppedImage);
            }
    
        }
    }
}



//json2dota
void test15(){

    std::string path = "/home/ubuntu/wjd/uav/jingbiao_dajiang/";

    std::string path2 = "/home/ubuntu/wjd/uav/dota_rail_fastener/";
    std::vector<std::string> json_names;

    listJpgFiles3(path,json_names);

    for(auto i : json_names){

        std::ifstream json_file(i); 
        json j = json::parse(json_file);

        int last_slash_pos = i.find_last_of("/\\");
        int last_dot_pos = i.find_last_of(".");
        std::string img_name = i.substr(last_slash_pos + 1, last_dot_pos - last_slash_pos - 1);
        std::string dota_name = path2 + img_name + ".txt";
        std::ofstream out(dota_name);

        // std::ofstream out_yolo("../yoloout/"+img_name+".txt");

        int width = j["imageHeight"];
        int height = j["imageWidth"];

        for(auto &item : j["shapes"]){
                
            std::string label_ = item["label"];

            // auto label_idx = std::find(class_names2.begin(),class_names2.end(),label_) - class_names2.begin();   

            if(label_=="fastener_normal"||label_=="fastener_abnormal"||label_=="fastener_stone"||label_=="fastener_missing" || label_ == "rail_big"){
                    
                if(item["points"].size()==4){

                    // out_yolo<<label_idx<<" ";
                    for(auto &ps: item["points"]){
                            out<<ps[0]<<" "<<ps[1]<<" ";
                            // out_yolo<<(float)ps[0]/width<<" "<<(float)ps[1]/height<<" ";

                    }
                    out<<label_<<" "<<"0\n";
                    // out_yolo<<"\n";
                }
            }

        } 
    
    }

}



#include "rotate.h"

// void test_single(){

//     std::ifstream f("..croped_dingzi/DJI_20240227105741_0001_1800_0.json");
//     json j = json::parse(f);

//     for(auto & i : j["shapes"]){

//         std::string label = i["label"];

//         std::vector<float> arr;
//         for(auto & ps : i["points"]){
//             float x = ps[0];
//             float y = ps[1];

//             arr.push_back(x);
//             arr.push_back(y);
//         }
//         for(int i = 0 ; i <arr.size() -2 ; i+=2){

//             std::cout<<"x =" << arr[i] << " y = " <<arr[i+1]<<std::endl;

//         }
  
//     }
// }



int main(){

    std::shared_ptr<Rotatation> rot = std::make_shared<Rotatation>();
    rot->run();
  
}