#include <iostream>
#include <filesystem>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const int SAMPLING_DENSITY = 30;
const float SAMPLE_WEIGHT = (1.0 / (3 * SAMPLING_DENSITY * SAMPLING_DENSITY));
const double THRESHOLD = 24;

const int RESOLUTION = 18;
int X_STEP;
int Y_STEP;
const int SCALE = 2;

vector<Mat> frames;
vector<Mat> ref_keyframes;
vector<Mat> render_keyframes;

/**
 * Tacks 0s on the beginning of a number until it reaches a specified number of digits.
 * @param i the number to pad.
 * @param the desired number of digits.
 * @return a string containing the padded value.
 */
String pad(int i, int length) {
    String padded = to_string(i);
    padded.insert(padded.begin(), length - padded.length(), '0');
    return padded;
}

/**
 * Compares two keyframes and returns their average difference.
 * @param m1 the first keyframe to be compared.
 * @param m2 the second keyframe to be compared.
 * @return the average difference at each pixel.
 */
double compare_frames(Mat m1, Mat m2) {
    if (m1.rows != m2.rows) return 0;
    if (m1.cols != m2.cols) return 0;

    int distance = 0;

    for(int row = 0; row < SAMPLING_DENSITY; row++) {
        for(int col = 0; col < SAMPLING_DENSITY; col++) {
            int x = col * m1.cols / SAMPLING_DENSITY;
            int y = row * m1.rows / SAMPLING_DENSITY;

            Vec3b pixel1 = m1.at<Vec3b>(y, x);
            Vec3b pixel2 = m2.at<Vec3b>(y, x);

            distance += abs(
                    pixel2[0] - pixel1[0] +
                    pixel2[1] - pixel1[1] +
                    pixel2[2] - pixel1[2]);
        }
    }

    return (float) distance * SAMPLE_WEIGHT;
}

/**
 * Removes all files matching a specified path.
 * @param path the path of the files to be removed.
 */
void clear_directory(const String& path) {
    cout << "Clearing " << path << "\n";

    vector<String> filenames;
    glob(path, filenames);

    for (const String& filename:filenames) {
        filesystem::remove(filename);
    }

    cout << "Directory cleared\n\n";
}

vector<String> get_all_filenames(const String& path) {
    vector<String> filenames;
    glob(path, filenames);

    return filenames;
}

/**
 * Returns all images matching a specified path.
 * @param path the path of the desired images.
 * @return a vector containing the images.
 */
vector<Mat> load_all(const String& path) {
    cout << "Reading in all from " << path << endl;

    vector<Mat> contents;
    vector<String> filenames = get_all_filenames(path);

    for (const String& filename: filenames) {
        contents.push_back(imread(filename));
    }

    cout << "Successfully read in " << contents.size() << " files.\n";
    return contents;
}

/**
 * Helper function picks out keyframes from a pool of frames and writes them to a cache directory.
 * @param frame_pool The pool of frames from which to choose keyframes from.
 * @return a vector containing the keyframes.
 */
vector<Mat> pick_keyframes(const vector<Mat>& frame_pool) {
    cout << "Generating new keyframes...";

    Mat previous_frame = frame_pool.at(0);
    Mat current_frame;

    vector<Mat> keyframes;
    keyframes.push_back(frame_pool.at(0));
    imwrite(
            "../src/keyframes/" + pad((int)keyframes.size(), 4) + ".png",
            frame_pool.at(0));

    for (const Mat& frame:frame_pool) {
        current_frame = frame;
        if (compare_frames(previous_frame, current_frame) > THRESHOLD) {
            keyframes.push_back(frame);

            imwrite("../src/keyframes/" + pad((int)keyframes.size(), 4) + ".png", frame);
            previous_frame = current_frame;
        }
    }
    cout << keyframes.size() << " keyframes identified.\n";

    return keyframes;
}

Mat get_frame_at(const String& filename) {
    return imread(filename);
}

void pick_keyframes(vector<String> filenames) {
    cout << "Picking keyframes...\n\n";

    Mat last_keyframe;
    int keyframe_counter = 0;

    // full black
    Mat this_frame = get_frame_at(filenames.at(0));
    imwrite(
            "../src/keyframes/" + pad(keyframe_counter, 4) + ".png",
            this_frame
    );
    last_keyframe = this_frame;

    // full white
    this_frame = get_frame_at(filenames.at(362));
    imwrite(
            "../src/keyframes/" + pad(++keyframe_counter, 4) + ".png",
            this_frame
    );

    int on_frame = 2;

    for (const String& filename:filenames) {
        on_frame++;

        this_frame = get_frame_at(filename);
        if (compare_frames(last_keyframe, this_frame) > THRESHOLD) {
            keyframe_counter++;

            imwrite("../src/keyframes/" + pad(keyframe_counter, 4) + ".png", this_frame);
            last_keyframe = this_frame;

            cout << "\033[FPicked " << keyframe_counter + 1 << " keyframes (" << (int)(100.0 * keyframe_counter / on_frame)  << "%)\n";
        }
    }
}

/**
 * Reads in keyframes from the cache if they exist, otherwise generates new ones.
 * @return a vector containing the keyframes.
 */
//vector<Mat> load_keyframes() {
//    cout << "Reading in frames...\n";
////    frames = load_all("../src/frames/*.png");
//    vector<Mat> existing_keyframes = load_all("../src/keyframes/*.png");
//
//    if (existing_keyframes.empty()) {
//        vector<String> filenames = get_all_filenames("../src/frames/*.png");
//        pick_keyframes(filenames);
//    }
//
//    cout << existing_keyframes.size() << " frames loaded.\n";
//    return existing_keyframes;
//}

/**
 * Scales a single keyframe to its reference size.
 * @param keyframe the keyframe to be scaled.
 */
void resize_keyframe(Mat keyframe) {
    const double scale = 1.0 / RESOLUTION;

    Mat ref_sized;
    Mat render_sized;

    resize(keyframe, ref_sized, Size(X_STEP, Y_STEP), INTER_LINEAR);
    resize(keyframe, render_sized, Size(X_STEP * SCALE, Y_STEP * SCALE), INTER_LINEAR);
    ref_keyframes.push_back(ref_sized);
    render_keyframes.push_back(render_sized);
}

/**
 * Scales all of the keyframes to their reference size.
 * @param keyframes the keyframes to be scaled.
 */
void resize_keyframes(vector<Mat> keyframes) {
    if (keyframes.empty()) return;
    const Mat _kf = keyframes.at(0);

    const double scale = 1.0 / RESOLUTION;
    X_STEP = ceil(scale * _kf.cols);
    Y_STEP = ceil(scale * _kf.rows);

    for (const auto& keyframe : keyframes) {
        resize_keyframe(_kf);
    }
}

void resize_keyframes(vector<String> filenames) {
    if (filenames.empty()) return;
    const Mat _kf = get_frame_at(filenames.at(0));

    const double scale = 1.0 / RESOLUTION;
    X_STEP = ceil(scale * _kf.cols);
    Y_STEP = ceil(scale * _kf.rows);

    int counter = 0;

    cout << endl;
    for (const String& filename : filenames) {
        cout << "\033[FResizing " << ++counter << "/" << filenames.size() << endl;
        Mat this_keyframe = get_frame_at(filename);
        resize_keyframe(this_keyframe);
    }
}

/**
 * Gets the keyframe index of the keyframe that best matches a specified rectangle of pixels.
 * @param frame_cell the pixels that each keyframe will be compared against.
 * @return the index of the closest keyframe.
 */
int get_closest_index(Mat frame_cell) {
    // Mat closest = ref_keyframes[0];
    int closest_index = 0;
    int distance = frame_cell.cols * frame_cell.rows * 255 * 3;

    for (int i = 0; i < ref_keyframes.size(); i++) {
        int this_distance = 0;
        for (int y = 0; y < frame_cell.rows; y++) {
            for (int x = 0; x < frame_cell.cols; x++){
                Vec3b actual = frame_cell.at<Vec3b>(y, x);
                Vec3b estimate = ref_keyframes[i].at<Vec3b>(y, x);

                this_distance += abs(
                        actual[0] - estimate[0] +
                        actual[1] - estimate[1] +
                        actual[2] - estimate[2]);
            }
        }
        if (this_distance >= distance) continue;
        // closest = keyframe;
        closest_index = i;
        distance = this_distance;
    }

    return closest_index;
}

/**
 * Renders an image by reconstructing it from the keyframes that most closely match each part.
 * @param frame the image to be rendered.
 * @return the reconstructed image..
 */
Mat render_frame(const Mat& frame) {
    Mat render = Mat::zeros(Size(frame.cols * SCALE, frame.rows * SCALE), CV_8UC3);

    for (int y = 0; y < RESOLUTION; y++) {

        for (int x = 0; x < RESOLUTION; x++) {
            double start_x  = (double)x/RESOLUTION * frame.cols;
            double start_y  = (double)y/RESOLUTION * frame.rows;

            int clamp_start_x = (int)round(start_x);
            int clamp_start_y = (int)round(start_y);

            Mat frame_roi = frame(Rect(
                    clamp_start_x, clamp_start_y,
                    min(X_STEP, frame.cols - clamp_start_x),
                    min(Y_STEP, frame.rows - clamp_start_y)
            ));
            int closest_index = get_closest_index(frame_roi);
            Mat closest = render_keyframes[closest_index];

            Mat dst_roi = render(Rect(
                    clamp_start_x * SCALE, clamp_start_y * SCALE,
                    min(X_STEP * SCALE, render.cols - clamp_start_x * SCALE),
                    min(Y_STEP * SCALE, render.rows - clamp_start_y * SCALE)
            ));
            Mat src_roi = closest(Rect(
                    0, 0,
                    min(X_STEP * SCALE, render.cols - clamp_start_x * SCALE),
                    min(Y_STEP * SCALE, render.rows - clamp_start_y * SCALE)
            ));
            src_roi.copyTo(dst_roi);
        }
    }

    return render;
}

/**
 * Renders each image in the global frames vector, starting from a specified index.
 * @param start the frame to start rendering from.
 */
void render(int start) {
    cout << "Rendering...\n";
    for (int i = start; i < frames.size(); i++) {
        Mat output = render_frame(frames[i]);
        imwrite("../output/render" + pad(i, 4) + ".png", output);
    }
    cout << "Render complete.\n";
}

void render(int start, const vector<String>& filenames) {
    cout << "Rendering...\n\n";
    for (int i = start; i < filenames.size(); i++) {
        clock_t start, end;

        cout << "\033[F\033[FRendering frame " << i << "/" << filenames.size() << endl;

        start = clock();

        Mat output = render_frame(imread(filenames.at(i)));
        imwrite("../output/render" + pad(i, 4) + ".png", output);

        end = clock();

        cout << "Last frame: " << difftime(end, start) / double(CLOCKS_PER_SEC) << endl;
    }
    cout << "Render complete.\n";
}

/**
 * Runs the program to take in a sequence of images and output each image, recreated using the sequence.
 * @param argc the number of supplied arguments.
 * @param argv the arguments supplied.
 */
int main(int argc, char *argv[]) {
    if ((String)argv[1] == "refresh") {
        cout << "Clearing old files...\n";

        clear_directory("../output/*.png");
        clear_directory("../src/keyframes/*.png");
    } else {
        cout << "Continuing from last run...\n";
    }

    cout << "Initializing program...\n";

    pick_keyframes(get_all_filenames("../src/frames/*.png"));
    cout << "Keyframes chosen.\n\n";

    cout << "Resizing keyframes...\n";
    resize_keyframes(get_all_filenames("../src/keyframes/*.png"));
    cout << "Keyframes resized.\n\n";

    vector<String> existing_output_frames;
    glob("../output/*.png", existing_output_frames);

    cout << "Found " << existing_output_frames.size() << " already rendered frames.\n";
    render((int)existing_output_frames.size(), get_all_filenames("../src/frames/*.png"));

    return 0;
}