// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file     model.cxx
///
/// \author   Christian Sonnabend <christian.sonnabend@cern.ch>
///
/// \brief    A general-purpose class with functions for ONNX model applications
///

// ONNX includes
#include "Tools/ML/ONNX/model.h"

namespace o2
{

namespace ml
{

std::string OnnxModel::printShape(const std::vector<int64_t>& v)
{
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

OnnxModel& OnnxModel::operator=(OnnxModel& inst)
{

  mEnv = inst.mEnv;
  mSession = inst.mSession;

  mInputNames = inst.mInputNames;
  mInputShapes = inst.mInputShapes;
  mOutputNames = inst.mOutputNames;
  mOutputShapes = inst.mOutputShapes;

  modelPath = inst.modelPath;
  ccdbUrl = inst.ccdbUrl;
  activeThreads = inst.activeThreads;
  valid_from = inst.valid_from;
  valid_until = inst.valid_until;

  LOG(debug) << "Model copied!";

  return *this;
}

bool OnnxModel::checkHyperloop()
{
  /// Testing hyperloop core settings
  const char* alien_cores = gSystem->Getenv("ALIEN_JDL_CPUCORES");
  bool alien_cores_found = (alien_cores != NULL);
  if (alien_cores_found) {
    LOGP(info, "Hyperloop test/Grid job detected! Number of cores = {}. Setting threads anyway to 1.", alien_cores);
    activeThreads = 1;
  } else {
    LOGP(info, "Not running on Hyperloop.");
  }

  return alien_cores_found;
}

OnnxModel::OnnxModel(std::string path)
{

  LOG(info) << "--- ONNX-ML model ---";

  modelPath = path;
  mEnv = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "onnx-model");
  mSession.reset(new Ort::Experimental::Session{*mEnv, path, sessionOptions});

  mInputNames = mSession->GetInputNames();
  mInputShapes = mSession->GetInputShapes();
  mOutputNames = mSession->GetOutputNames();
  mOutputShapes = mSession->GetOutputShapes();

  LOG(info) << "Input Nodes:";
  for (size_t i = 0; i < mInputNames.size(); i++) {
    LOG(info) << "\t" << mInputNames[i] << " : " << printShape(mInputShapes[i]);
  }

  LOG(info) << "Output Nodes:";
  for (size_t i = 0; i < mOutputNames.size(); i++) {
    LOG(info) << "\t" << mOutputNames[i] << " : " << printShape(mOutputShapes[i]);
  }

  LOG(info) << "--- Model initialized! ---";

  if (checkHyperloop()) {
    sessionOptions.SetIntraOpNumThreads(activeThreads);
  }
}

bool OnnxModel::fetchFromCCDB(std::string path_from, int64_t ccdbTimestamp, std::string path_to = "model.onnx")
{

  LOG(info) << "--- ONNX-ML model ---";

  // e.g. path_from = "Analysis/PID/TPC/ML"

  o2::ccdb::CcdbApi ccdbApi;
  ccdbApi.init(ccdbUrl);

  std::map<std::string, std::string> metadata;
  bool retrieve_success = ccdbApi.retrieveBlob(path_from, ".", metadata, ccdbTimestamp, false, path_to);
  std::map<std::string, std::string> headers = ccdbApi.retrieveHeaders(path_from, metadata, ccdbTimestamp);

  if (headers.count("Valid-From") == 0) {
    LOG(info) << "Valid-From not found in metadata";
  } else {
    valid_until = strtoul(headers["Valid-From"].c_str(), NULL, 0);
  }
  if (headers.count("Valid-Until") == 0) {
    LOG(info) << "Valid-Until not found in metadata";
  } else {
    strtoul(headers["Valid-Until"].c_str(), NULL, 0);
  }

  if (checkHyperloop()) {
    sessionOptions.SetIntraOpNumThreads(activeThreads);
  }

  mEnv = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "onnx-model");
  mSession.reset(new Ort::Experimental::Session{*mEnv, path_to, sessionOptions});

  return retrieve_success;
}

bool OnnxModel::downloadToFile(std::string path_from, int64_t ccdbTimestamp, std::string path_to = "model.onnx")
{

  // e.g. path_from = "Analysis/PID/TPC/ML"

  o2::ccdb::CcdbApi ccdbApi;
  ccdbApi.init(ccdbUrl);

  std::map<std::string, std::string> metadata;
  bool retrieve_success = ccdbApi.retrieveBlob(path_from, ".", metadata, ccdbTimestamp, false, path_to);
  std::map<std::string, std::string> headers = ccdbApi.retrieveHeaders(path_from, metadata, ccdbTimestamp);

  if (headers.count("Valid-From") == 0) {
    LOG(info) << "Valid-From not found in metadata";
  } else {
    LOG(info) << "Timestamp, valid from: " << headers["Valid-From"].c_str();
  }
  if (headers.count("Valid-Until") == 0) {
    LOG(info) << "Valid-Until not found in metadata";
  } else {
    LOG(info) << "Timestamp, valid until: " << headers["Valid-Until"].c_str();
  }

  return retrieve_success;
}

float* OnnxModel::evalModel(std::vector<Ort::Value> input)
{

  try {
    LOG(debug) << "Shape of input (tensor): " << printShape(input[0].GetTensorTypeAndShapeInfo().GetShape());

    auto outputTensors = mSession->Run(mInputNames, input, mOutputNames);
    float* output_values = outputTensors[0].GetTensorMutableData<float>();
    LOG(debug) << "Shape of output (tensor): " << printShape(outputTensors[0].GetTensorTypeAndShapeInfo().GetShape());

    return output_values;

  } catch (const Ort::Exception& exception) {
    LOG(error) << "Error running model inference: " << exception.what();

    return 0;
  }
}

float* OnnxModel::evalModel(std::vector<float> input)
{

  int64_t size = input.size();
  std::vector<int64_t> input_shape{size / mInputShapes[0][1], mInputShapes[0][1]};
  std::vector<Ort::Value> inputTensors;
  inputTensors.emplace_back(Ort::Experimental::Value::CreateTensor<float>(input.data(), size, input_shape));

  try {

    LOG(debug) << "Shape of input (vector): " << printShape(input_shape);
    auto outputTensors = mSession->Run(mInputNames, inputTensors, mOutputNames);
    LOG(debug) << "Shape of output (tensor): " << printShape(outputTensors[0].GetTensorTypeAndShapeInfo().GetShape());
    float* output_values = outputTensors[0].GetTensorMutableData<float>();

    return output_values;

  } catch (const Ort::Exception& exception) {

    LOG(error) << "Error running model inference: " << exception.what();

    return 0;
  }
}

} // namespace ml

} // namespace o2