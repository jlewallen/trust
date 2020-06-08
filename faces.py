#!/usr/bin/python3

import logging
import argparse
import sys
import os
import fnmatch
import json
import jsonpickle

import dlib
import joblib
import cv2
import cvlib
import face_recognition

HearCascadePath = "haarcascade_frontalface_default.xml"


class FileLocation:
    def __init__(self, path):
        self.path = path
        self.dir = os.path.dirname(path)
        self.name = os.path.basename(path)

    def path_for_sidecar_file(self, extension):
        return os.path.join(self.dir, self.name + "." + extension)

    def path_for_sidecar_jpg(self, extension):
        p = os.path.splitext(self.name)
        return os.path.join(self.dir, p[0] + "." + extension + p[1])


class Image:
    def __init__(self, path):
        self.path = path

    def analyse(self):
        pass

    def write_meta(self, image_path, meta):
        meta_path = FileLocation(image_path).path_for_sidecar_file("meta")
        with open(meta_path, "w") as file:
            file.write(jsonpickle.encode(meta))

    def read_meta(self, image_path):
        meta_path = FileLocation(image_path).path_for_sidecar_file("meta")
        try:
            with open(meta_path, "r") as file:
                return jsonpickle.decode(file.read())
        except IOError:
            return None


class ImageMeta:
    def __init__(self, d):
        self.d = d


class FaceInstance:
    def __init__(self, location, encoding):
        self.location = location
        self.encoding = encoding

    def __str__(self):
        return "<%s>" % (self.location,)

    def __repr__(self):
        return self.__str__()


class Classifier:
    def __init__(self, nthreads=1):
        self.nthreads = nthreads
        self.log = logging.getLogger("faces")
        self.storage_path = "./processed"
        self.hearCascadePath = "haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(self.hearCascadePath)

    def get_candidate_files(self, paths):
        matches = []
        for path in paths:
            for file_dir, dirs, files in os.walk(path):
                for file in fnmatch.filter(files, "*.jpg"):
                    if fnmatch.fnmatch(file, "*.faces.jpg"):
                        continue
                    if fnmatch.fnmatch(file, "*.cnn.jpg"):
                        continue
                    if fnmatch.fnmatch(file, "*.haar.jpg"):
                        continue
                    matches.append(os.path.join(file_dir, file))
        return matches

    def include_paths(self, paths):
        candidates = self.get_candidate_files(paths)
        if self.nthreads > 1:
            return joblib.Parallel(n_jobs=self.nthreads, prefer="threads")(
                joblib.delayed(self.include_file)(path) for path in candidates
            )
        return [self.include_file(path) for path in candidates]

    def analyse_haar(self, path, original, annotated):
        self.log.info("%s haar:begin" % (path,))

        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) > 0:
            annotated = original.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            haar_path = FileLocation(path).path_for_sidecar_jpg("haar")
            cv2.imwrite(haar_path, annotated)

        self.log.info("%s haar:faces=%s" % (path, faces))

    def analyse_cvlib(self, path, original, annotated):
        self.log.info("%s cvlib:begin" % (path,))

        meta = cvlib.detect_face(original)

        self.log.info("%s cvlib:faces=%s" % (path, meta))

        return meta

    def analyse_cnn(self, path, original, annotated):
        self.log.info("%s fd-cnn:begin" % (path,))

        small_image = cv2.resize(original, (0, 0), fx=0.50, fy=0.50)
        locations = face_recognition.face_locations(small_image, model="cnn")
        encodings = face_recognition.face_encodings(small_image, locations, model="cnn")

        faces = [FaceInstance(l, e) for (l, e) in zip(locations, encodings)]

        # We only write a faces file if there's actually faces.
        if len(faces) > 0:
            for (y1, x1, y2, x2) in locations:
                cv2.rectangle(small_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cnn_path = FileLocation(path).path_for_sidecar_jpg("cnn")
            cv2.imwrite(cnn_path, small_image)

        self.log.info("%s fd-cnn:faces=%s" % (path, faces))

        return faces

    def include_file(self, path):
        image = Image(path)

        existing_meta = image.read_meta(path)
        if existing_meta:
            self.log.info("%s skipping, cached" % (path,))
            return existing_meta

        self.log.info("%s loading" % (path,))

        original = face_recognition.load_image_file(path)

        cvlib = self.analyse_cvlib(path, original, None)
        haar = self.analyse_haar(path, original, None)
        cnn = self.analyse_cnn(path, original, None)

        self.log.info("%s saving" % (path,))

        meta = ImageMeta({"cvlib": cvlib, "haar": haar, "cnn": cnn})
        image.write_meta(path, meta)

        return meta


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    log = logging.getLogger("faces")

    parser = argparse.ArgumentParser(description="faces")
    parser.add_argument("-n", "--no-cache", action="store_true", default=False)
    parser.add_argument("-k", "--check", action="store_true", default=False)
    parser.add_argument("-j", "--threads", action="store", type=int, default=1)
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()

    if args.check:
        log.info("%s %d" % (dlib.DLIB_USE_CUDA, dlib.cuda.get_num_devices()))

    if args.paths:
        log.info("starting (threads=%d)" % (args.threads,))
        classifier = Classifier(args.threads)
        classifier.include_paths(args.paths)
        log.info("done")
