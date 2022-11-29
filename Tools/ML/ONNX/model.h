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
/// \file     model.h
///
/// \author   Christian Sonnabend <christian.sonnabend@cern.ch>
///
/// \brief    A general-purpose class for ONNX models
///

#ifndef TOOLS_ML_ONNX_MODEL_H_
#define TOOLS_ML_ONNX_MODEL_H_

// C++ and system includes
#include <onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <map>

// ROOT includes
#include "TSystem.h"

// O2 includes
#include "CCDB/CcdbApi.h"
#include "Framework/Logger.h"

namespace o2
{

namespace ml
{

class OnnxModel
{

 public:
  OnnxModel() = default;
  explicit OnnxModel(std::string);
  ~OnnxModel() = default;
  OnnxModel& operator=(OnnxModel&);

  // IO
  bool fetchFromCCDB(std::string, int64_t, std::string);
  bool downloadToFile(std::string, int64_t, std::string);
  float* evalModel(std::vector<Ort::Value>);
  float* evalModel(std::vector<float>);

  // Optimizations
  Ort::SessionOptions* getSessionOptions() { return &sessionOptions; }
  void resetSession() { mSession.reset(new Ort::Experimental::Session{*mEnv, modelPath, sessionOptions}); }

  // Getters & Setters
  int getInputDimensions() const { return mInputShapes[0][1]; }
  int getOutputDimensions() const { return mOutputShapes[0][1]; }
  uint64_t getValidityFrom() const { return valid_from; }
  uint64_t getValidityUntil() const { return valid_until; }
  void setCcdbUrl(std::string url) { ccdbUrl = url; }
  void setActiveThreads(int threads) { activeThreads = threads; }

 private:
  // Environment variables for the ONNX runtime
  std::shared_ptr<Ort::Env> mEnv = nullptr;
  std::shared_ptr<Ort::Experimental::Session> mSession = nullptr;
  Ort::SessionOptions sessionOptions;

  // Input & Output specifications of the loaded network
  std::vector<std::string> mInputNames;
  std::vector<std::vector<int64_t>> mInputShapes;
  std::vector<std::string> mOutputNames;
  std::vector<std::vector<int64_t>> mOutputShapes;

  // Environment settings
  std::string modelPath;
  std::string ccdbUrl = "http://alice-ccdb.cern.ch";
  int activeThreads = 0;
  int64_t valid_from = -1;
  int64_t valid_until = -1;

  // Internal function for printing the shape of tensors
  std::string printShape(const std::vector<int64_t>& v);
  bool checkHyperloop();
};

} // namespace ml

} // namespace o2

#endif // TOOLS_ML_ONNX_MODEL_H_
