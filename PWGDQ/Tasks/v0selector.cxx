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
//
// Example analysis task to select clean V0 sample
// ========================
//
// This code loops over a V0Data table and produces some standard analysis output.
//
//    Comments, questions, complaints, suggestions?
//    Please write to: daiki.sekihata@cern.ch
//
#include "PWGDQ/Core/HistogramManager.h"
#include "PWGDQ/Core/HistogramsLibrary.h"
#include "PWGDQ/Core/VarManager.h"
#include "PWGDQ/DataModel/ReducedInfoTables.h"
#include "PWGLF/DataModel/LFStrangenessTables.h"

#include "Common/Core/RecoDecay.h"
#include "Common/Core/TrackSelection.h"
#include "Common/Core/trackUtilities.h"
#include "Common/DataModel/Centrality.h"
#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/PIDResponse.h"
#include "Common/DataModel/TrackSelectionTables.h"

#include "CCDB/BasicCCDBManager.h"
#include "DCAFitter/DCAFitterN.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "Framework/ASoAHelpers.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisTask.h"
#include "Framework/runDataProcessing.h"
#include "ReconstructionDataFormats/Track.h"

#include "Math/Vector4D.h"

#include <array>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <sstream>
#include <unordered_map>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

/*using FullTracksExt = soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra, aod::TracksDCA,
                                aod::pidTPCFullEl, aod::pidTPCFullPi,
                                aod::pidTPCFullKa, aod::pidTPCFullPr>;*/
using FullTracksExt = soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksDCA,
                                aod::pidTPCFullEl, aod::pidTPCFullPi,
                                aod::pidTPCFullKa, aod::pidTPCFullPr>;

constexpr static uint32_t gkTrackFillMap = VarManager::ObjTypes::Track | VarManager::ObjTypes::TrackExtra | VarManager::ObjTypes::TrackTPCPID;

struct v0selector {

  // Configurables for curved QT cut
  //  Gamma cuts
  Configurable<float> cutAlphaG{"cutAlphaG", 0.4, "cutAlphaG"};
  Configurable<float> cutQTG{"cutQTG", 0.03, "cutQTG"};
  Configurable<float> cutAlphaGLow{"cutAlphaGLow", 0.4, "cutAlphaGLow"};
  Configurable<float> cutAlphaGHigh{"cutAlphaGHigh", 0.8, "cutAlphaGHigh"};
  Configurable<float> cutQTG2{"cutQTG2", 0.02, "cutQTG2"};
  // K0S cuts
  Configurable<float> cutQTK0SLow{"cutQTK0SLow", 0.1075, "cutQTK0SLow"};
  Configurable<float> cutQTK0SHigh{"cutQTK0SHigh", 0.215, "cutQTK0SHigh"};
  Configurable<float> cutAPK0SLow{"cutAPK0SLow", 0.199, "cutAPK0SLow"};
  Configurable<float> cutAPK0SHigh{"cutAPK0SHigh", 0.8, "cutAPK0SHigh"};
  Configurable<float> cutAPK0SHighTop{"cutAPK0SHighTop", 0.9, "cutAPK0SHighTop"};
  // Lambda & A-Lambda cuts
  Configurable<float> cutQTL{"cutQTL", 0.03, "cutQTL"};
  Configurable<float> cutAlphaLLow{"cutAlphaLLow", 0.35, "cutAlphaLLow"};
  Configurable<float> cutAlphaLHigh{"cutAlphaLHigh", 0.7, "cutAlphaLHigh"};
  Configurable<float> cutAlphaALLow{"cutAlphaALow", -0.7, "cutAlphaALow"};
  Configurable<float> cutAlphaALHigh{"cutAlphaAHigh", -0.35, "cutAlphaAHigh"};
  Configurable<float> cutAPL1{"cutAPL1", 0.107, "cutAPL1"};
  Configurable<float> cutAPL2{"cutAPL2", -0.69, "cutAPL2"};
  Configurable<float> cutAPL3{"cutAPL3", 0.5, "cutAPL3"};
  // Omega & A-Omega cuts
  Configurable<float> cutAPOmegaUp1{"cutAPOmegaUp1", 0.25, "cutAPOmegaUp1"};
  Configurable<float> cutAPOmegaUp2{"cutAPOmegaUp2", 0.358, "cutAPOmegaUp2"};
  Configurable<float> cutAPOmegaUp3{"cutAPOmegaUp3", 0.35, "cutAPOmegaUp3"};
  Configurable<float> cutAPOmegaDown1{"cutAPOmegaDown1", 0.15, "cutAPOmegaDown1"};
  Configurable<float> cutAPOmegaDown2{"cutAPOmegaDown2", 0.358, "cutAPOmegaDown2"};
  Configurable<float> cutAPOmegaDown3{"cutAPOmegaDown3", 0.16, "cutAPOmegaDown3"};
  Configurable<float> cutAlphaOmegaHigh{"cutAlphaOmegaHigh", 0.358, "cutAlphaOmegaHigh"};
  Configurable<float> cutAlphaOmegaLow{"cutAlphaOmegaLow", 0., "cutAlphaOmegaLow"};
  Configurable<float> cutQTOmegaLowOuterArc{"cutQTOmegaLowOuterArc", 0.14, "cutQTOmegaLowOuterArc"};
  Configurable<float> cutMassOmegaHigh{"cutMassOmegaHigh", 1.677, "cutMassOmegaHigh"};
  Configurable<float> cutMassOmegaLow{"cutMassOmegaLow", 1.667, "cutMassOmegaLow"};
  Configurable<float> cutMassCascV0Low{"cutMassCascV0Low", 1.110, "cutMassCascV0Low"};
  Configurable<float> cutMassCascV0High{"cutMassCascV0High", 1.120, "cutMassCascV0High"};

  Configurable<bool> produceV0ID{"produceV0ID", false, "Produce additional V0ID table"};
  Configurable<bool> selectCascades{"selectCascades", false, "Select cascades in addition to v0s"};
  Configurable<bool> produceCascID{"produceCascID", false, "Produce additional CascID table"};

  enum { // Reconstructed V0
    kUndef = -1,
    kGamma = 0,
    kK0S = 1,
    kLambda = 2,
    kAntiLambda = 3,
    kOmega = 4,
    kAntiOmega = 5
  };

  Produces<o2::aod::V0Bits> v0bits;
  Produces<o2::aod::V0MapID> v0mapID;
  Produces<o2::aod::CascMapID> cascmapID;

  // int checkV0(const array<float, 3>& ppos, const array<float, 3>& pneg)
  int checkV0(const float alpha, const float qt)
  {
    // float alpha = alphav0(ppos, pneg);
    // float qt = qtarmv0(ppos, pneg);
    // // Gamma cuts
    // const float cutAlphaG = 0.4;
    // const float cutQTG = 0.03;
    // const float cutAlphaG2[2] = {0.4, 0.8};
    // const float cutQTG2 = 0.02;

    // // K0S cuts
    // const float cutQTK0S[2] = {0.1075, 0.215};
    // const float cutAPK0S[2] = {0.199, 0.8}; // parameters for curved QT cut

    // // Lambda & A-Lambda cuts
    // const float cutQTL = 0.03;
    // const float cutAlphaL[2] = {0.35, 0.7};
    // const float cutAlphaAL[2] = {-0.7, -0.35};
    // const float cutAPL[3] = {0.107, -0.69, 0.5}; // parameters for curved QT cut

    if (qt < cutQTG) {
      if ((std::abs(alpha) < cutAlphaG)) {
        return kGamma;
      }
    }
    if (qt < cutQTG2) {
      // additional region - should help high pT gammas
      if ((std::abs(alpha) > cutAlphaGLow) && (std::abs(alpha) < cutAlphaGHigh)) {
        return kGamma;
      }
    }

    // Check for K0S candidates
    float q = cutAPK0SLow * std::sqrt(std::abs(1.0f - alpha * alpha / (cutAPK0SHigh * cutAPK0SHigh)));
    float qtop = cutQTK0SHigh * std::sqrt(std::abs(1.0f - alpha * alpha / (cutAPK0SHighTop * cutAPK0SHighTop)));
    if ((qt > cutQTK0SLow) && (qt < cutQTK0SHigh) && (qt > q) && (qt < qtop)) {
      return kK0S;
    }

    // Check for Lambda candidates
    q = cutAPL1 * std::sqrt(std::abs(1.0f - ((alpha + cutAPL2) * (alpha + cutAPL2)) / (cutAPL3 * cutAPL3)));
    if ((alpha > cutAlphaLLow) && (alpha < cutAlphaLHigh) && (qt > cutQTL) && (qt < q)) {
      return kLambda;
    }

    // Check for AntiLambda candidates
    q = cutAPL1 * std::sqrt(std::abs(1.0f - ((alpha - cutAPL2) * (alpha - cutAPL2)) / (cutAPL3 * cutAPL3)));
    if ((alpha > cutAlphaALLow) && (alpha < cutAlphaALHigh) && (qt > cutQTL) && (qt < q)) {
      return kAntiLambda;
    }

    return kUndef;
  }

  int checkCascade(float alpha, float qt)
  {
    const bool isAlphaPos = alpha > 0;
    alpha = std::fabs(alpha);

    const float qUp = std::abs(alpha - cutAPOmegaUp2) > std::abs(cutAPOmegaUp3) ? 0. : cutAPOmegaUp1 * std::sqrt(1.0f - ((alpha - cutAPOmegaUp2) * (alpha - cutAPOmegaUp2)) / (cutAPOmegaUp3 * cutAPOmegaUp3));
    const float qDown = std::abs(alpha - cutAPOmegaDown2) > std::abs(cutAPOmegaDown3) ? 0. : cutAPOmegaDown1 * std::sqrt(1.0f - ((alpha - cutAPOmegaDown2) * (alpha - cutAPOmegaDown2)) / (cutAPOmegaDown3 * cutAPOmegaDown3));

    if (alpha < cutAlphaOmegaLow || alpha > cutAlphaOmegaHigh || qt < qDown || qt > qUp) {
      return kUndef;
    }

    if (alpha > cutAPOmegaUp2 && qt < cutQTOmegaLowOuterArc) {
      return kUndef;
    }

    if (isAlphaPos) {
      return kOmega;
    } else {
      return kAntiOmega;
    }
  }

  // Configurables
  Configurable<float> v0max_mee{"v0max_mee", 0.1, "max mee for photon"};
  Configurable<float> maxpsipair{"maxpsipair", 1.6, "max psi_pair for photon"};
  Configurable<float> v0cospa{"v0cospa", 0.998, "V0 CosPA"};
  Configurable<float> dcav0dau{"dcav0dau", 0.3, "DCA V0 Daughters"};
  Configurable<float> v0Rmin{"v0Rmin", 0.0, "v0Rmin"};
  Configurable<float> v0Rmax{"v0Rmax", 90.0, "v0Rmax"};
  Configurable<float> dcamin{"dcamin", 0.0, "dcamin"};
  Configurable<float> dcamax{"dcamax", 1e+10, "dcamax"};
  Configurable<int> mincrossedrows{"mincrossedrows", 70, "min crossed rows"};
  Configurable<float> maxchi2tpc{"maxchi2tpc", 4.0, "max chi2/NclsTPC"};
  Configurable<bool> fillhisto{"fillhisto", false, "flag to fill histograms"};
  // Cascade-related Configurables
  Configurable<float> cascDcaMax{"cascDcaMax", 0.3, "DCA cascade daughters"};
  Configurable<float> cascV0DcaMax{"cascV0DcaMax", 0.3, "DCA V0 daughters of the cascade"};
  Configurable<float> cascRadiusMin{"cascRadiusMin", 0.0, "Cascade Radius min"};
  Configurable<float> cascRadiusMax{"cascRadiusMax", 90.0, "Cascade Radius max"};
  Configurable<float> cascV0RadiusMin{"cascV0RadiusMin", 0.0, "V0 of the Cascade Radius min"};
  Configurable<float> cascV0RadiusMax{"cascV0RadiusMax", 90.0, "V0 of the Cascade Radius max"};
  Configurable<float> cascCosinePAMin{"cascCosinePAMin", 0.998, "Cascade CosPA min"};
  Configurable<float> cascV0CosinePAMin{"cascV0CosinePAMin", 0.995, "V0 of the Cascade CosPA min"};
  Configurable<float> cascV0CosinePAMax{"cascV0CosinePAMax", 1.000, "V0 of the Cascade CosPA max"};
  // cutNsigmaElTPC, cutNsigmaPiTPC, cutNsigmaPrTPC, cutNsigmaKaTPC
  Configurable<float> cutNsigmaElTPC{"cutNsigmaElTPC", 5.0, "cutNsigmaElTPC"};
  Configurable<float> cutNsigmaPiTPC{"cutNsigmaPiTPC", 5.0, "cutNsigmaPiTPC"};
  Configurable<float> cutNsigmaPrTPC{"cutNsigmaPrTPC", 5.0, "cutNsigmaPrTPC"};
  Configurable<float> cutNsigmaKaTPC{"cutNsigmaKaTPC", 5.0, "cutNsigmaKaTPC"};

  // Consolidated histogram binning configuration: to avoid exceeding framework aggregate limits
  Configurable<std::string> v0BinningConfig{"v0BinningConfig", "", "Override V0/cascade histogram binning as comma-separated key=value list (e.g. 'nBinsRadius=900,maxRadius=90'). Only provided keys are overridden."};

  struct V0BinningParams {
    int nBinsV0Candidate = 2; float minV0Candidate = 0.5f; float maxV0Candidate = 2.5f;
    int nBinsRadius = 900; float minRadius = 0.f; float maxRadius = 90.f;
    int nBinsMassGamma = 100; float minMassGamma = 0.f; float maxMassGamma = 0.1f;
    int nBinsGammaRxy = 1800; float minGammaRxy = -90.f; float maxGammaRxy = 90.f;
    int nBinsMassK0S = 100; float minMassK0S = 0.45f; float maxMassK0S = 0.55f;
    int nBinsPt = 200; float minPt = 0.f; float maxPtK0S = 20.f;
    int nBinsEta = 20; float minEta = -1.f; float maxEta = 1.f;
    int nBinsPhi = 63; float minPhi = 0.f; float maxPhi = 6.3f;
    int nBinsMassLambda = 100; float minMassLambda = 1.05f; float maxMassLambda = 1.15f;
    int nBinsV0Pt = 100; float maxV0Pt = 10.f;
    int nBinsV0Radius = 1000; float maxV0Radius = 100.f;
    int nBinsV0CosPA = 50; float minV0CosPA = 0.95f; float maxV0CosPA = 1.0f;
    int nBinsDCA = 1000; float minDCA = -5.f; float maxDCA = 5.f;
    int nBinsDCAV0Dau = 1000; float maxDCAV0Dau = 10.f;
    int nBinsAPAlpha = 200; float minAPAlpha = -1.f; float maxAPAlpha = 1.f;
    int nBinsAPQt = 250; float minAPQt = 0.f; float maxAPQt = 0.25f;
    int nBinsV0Psi = 100; float maxV0Psi = 0.1f;
    // Cascade specific
    int nBinsCascPt = 100; float maxCascPt = 10.f;
    int nBinsCascAPQt = 300; float maxCascAPQt = 0.3f;
    int nBinsCascRadius = 1000; float maxCascRadius = 100.f;
    int nBinsMassOmega = 100; float minMassOmega = 1.62f; float maxMassOmega = 1.72f;

    void applyOverrides(const std::string& cfg)
    {
      if (cfg.empty()) return;
      std::unordered_map<std::string, std::string> kv;
      std::stringstream ss(cfg);
      std::string token;
      while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        auto eq = token.find('=');
        if (eq == std::string::npos) continue;
        std::string key = token.substr(0, eq);
        std::string val = token.substr(eq + 1);
        // trim spaces
        auto trim = [](std::string& s){ while(!s.empty() && isspace(s.front())) s.erase(s.begin()); while(!s.empty() && isspace(s.back())) s.pop_back();};
        trim(key); trim(val);
        if (!key.empty()) kv[key] = val;
      }
      auto setInt=[&](const char* name, int& ref){ auto it=kv.find(name); if(it!=kv.end()) ref=std::stoi(it->second);};
      auto setF=[&](const char* name, float& ref){ auto it=kv.find(name); if(it!=kv.end()) ref=std::stof(it->second);};
      setInt("nBinsV0Candidate", nBinsV0Candidate); setF("minV0Candidate", minV0Candidate); setF("maxV0Candidate", maxV0Candidate);
      setInt("nBinsRadius", nBinsRadius); setF("minRadius", minRadius); setF("maxRadius", maxRadius);
      setInt("nBinsMassGamma", nBinsMassGamma); setF("minMassGamma", minMassGamma); setF("maxMassGamma", maxMassGamma);
      setInt("nBinsGammaRxy", nBinsGammaRxy); setF("minGammaRxy", minGammaRxy); setF("maxGammaRxy", maxGammaRxy);
      setInt("nBinsMassK0S", nBinsMassK0S); setF("minMassK0S", minMassK0S); setF("maxMassK0S", maxMassK0S);
      setInt("nBinsPt", nBinsPt); setF("minPt", minPt); setF("maxPtK0S", maxPtK0S);
      setInt("nBinsEta", nBinsEta); setF("minEta", minEta); setF("maxEta", maxEta);
      setInt("nBinsPhi", nBinsPhi); setF("minPhi", minPhi); setF("maxPhi", maxPhi);
      setInt("nBinsMassLambda", nBinsMassLambda); setF("minMassLambda", minMassLambda); setF("maxMassLambda", maxMassLambda);
      setInt("nBinsV0Pt", nBinsV0Pt); setF("maxV0Pt", maxV0Pt);
      setInt("nBinsV0Radius", nBinsV0Radius); setF("maxV0Radius", maxV0Radius);
      setInt("nBinsV0CosPA", nBinsV0CosPA); setF("minV0CosPA", minV0CosPA); setF("maxV0CosPA", maxV0CosPA);
      setInt("nBinsDCA", nBinsDCA); setF("minDCA", minDCA); setF("maxDCA", maxDCA);
      setInt("nBinsDCAV0Dau", nBinsDCAV0Dau); setF("maxDCAV0Dau", maxDCAV0Dau);
      setInt("nBinsAPAlpha", nBinsAPAlpha); setF("minAPAlpha", minAPAlpha); setF("maxAPAlpha", maxAPAlpha);
      setInt("nBinsAPQt", nBinsAPQt); setF("minAPQt", minAPQt); setF("maxAPQt", maxAPQt);
      setInt("nBinsV0Psi", nBinsV0Psi); setF("maxV0Psi", maxV0Psi);
      setInt("nBinsCascPt", nBinsCascPt); setF("maxCascPt", maxCascPt);
      setInt("nBinsCascAPQt", nBinsCascAPQt); setF("maxCascAPQt", maxCascAPQt);
      setInt("nBinsCascRadius", nBinsCascRadius); setF("maxCascRadius", maxCascRadius);
      setInt("nBinsMassOmega", nBinsMassOmega); setF("minMassOmega", minMassOmega); setF("maxMassOmega", maxMassOmega);
    }
  } v0Bins; // instance


  HistogramRegistry registry{"registry"};
  void init(o2::framework::InitContext&)
  {
    // Apply overrides for consolidated binning parameters
    v0Bins.applyOverrides(v0BinningConfig.value);

    if (fillhisto) {
      const auto& b = v0Bins;
      registry.add("hV0Candidate", "hV0Candidate", HistType::kTH1F, {{b.nBinsV0Candidate, b.minV0Candidate, b.maxV0Candidate}});
      registry.add("hCascCandidate", "hCascCandidate", HistType::kTH1F, {{b.nBinsV0Candidate, b.minV0Candidate, b.maxV0Candidate}});
      registry.add("hMassGamma", "hMassGamma", HistType::kTH2F, {{b.nBinsRadius, b.minRadius, b.maxRadius}, {b.nBinsMassGamma, b.minMassGamma, b.maxMassGamma}});
      registry.add("hGammaRxy", "hGammaRxy", HistType::kTH2F, {{b.nBinsGammaRxy, b.minGammaRxy, b.maxGammaRxy}, {b.nBinsGammaRxy, b.minGammaRxy, b.maxGammaRxy}});
      registry.add("hMassK0S", "hMassK0S", HistType::kTH2F, {{b.nBinsRadius, b.minRadius, b.maxRadius}, {b.nBinsMassK0S, b.minMassK0S, b.maxMassK0S}});
      registry.add("hMassK0SPt", "hMassK0SPt", HistType::kTH2F, {{b.nBinsPt, b.minPt, b.maxPtK0S}, {b.nBinsMassK0S, b.minMassK0S, b.maxMassK0S}});
      registry.add("hMassK0SEta", "hMassK0SEta", HistType::kTH2F, {{b.nBinsEta, b.minEta, b.maxEta}, {b.nBinsMassK0S, b.minMassK0S, b.maxMassK0S}});
      registry.add("hMassK0SPhi", "hMassK0SPhi", HistType::kTH2F, {{b.nBinsPhi, b.minPhi, b.maxPhi}, {b.nBinsMassK0S, b.minMassK0S, b.maxMassK0S}});
      registry.add("hMassLambda", "hMassLambda", HistType::kTH2F, {{b.nBinsRadius, b.minRadius, b.maxRadius}, {b.nBinsMassLambda, b.minMassLambda, b.maxMassLambda}});
      registry.add("hMassAntiLambda", "hAntiMassLambda", HistType::kTH2F, {{b.nBinsRadius, b.minRadius, b.maxRadius}, {b.nBinsMassLambda, b.minMassLambda, b.maxMassLambda}});
      registry.add("hV0Pt", "pT", HistType::kTH1F, {{b.nBinsV0Pt, b.minPt, b.maxV0Pt}});
      registry.add("hV0EtaPhi", "#eta vs. #varphi", HistType::kTH2F, {{b.nBinsPhi, b.minPhi, b.maxPhi}, {b.nBinsEta, b.minEta, b.maxEta}});
      registry.add("hV0Radius", "hV0Radius", HistType::kTH1F, {{b.nBinsV0Radius, b.minPt, b.maxV0Radius}});
      registry.add("hV0CosPA", "hV0CosPA", HistType::kTH1F, {{b.nBinsV0CosPA, b.minV0CosPA, b.maxV0CosPA}});
      registry.add("hDCAxyPosToPV", "hDCAxyPosToPV", HistType::kTH1F, {{b.nBinsDCA, b.minDCA, b.maxDCA}});
      registry.add("hDCAxyNegToPV", "hDCAxyNegToPV", HistType::kTH1F, {{b.nBinsDCA, b.minDCA, b.maxDCA}});
      registry.add("hDCAzPosToPV", "hDCAzPosToPV", HistType::kTH1F, {{b.nBinsDCA, b.minDCA, b.maxDCA}});
      registry.add("hDCAzNegToPV", "hDCAzNegToPV", HistType::kTH1F, {{b.nBinsDCA, b.minDCA, b.maxDCA}});
      registry.add("hDCAV0Dau", "hDCAV0Dau", HistType::kTH1F, {{b.nBinsDCAV0Dau, b.minPt, b.maxDCAV0Dau}});
      registry.add("hV0APplot", "hV0APplot", HistType::kTH2F, {{b.nBinsAPAlpha, b.minAPAlpha, b.maxAPAlpha}, {b.nBinsAPQt, b.minAPQt, b.maxAPQt}});
      registry.add("hV0APplotSelected", "hV0APplotSelected", HistType::kTH2F, {{b.nBinsAPAlpha, b.minAPAlpha, b.maxAPAlpha}, {b.nBinsAPQt, b.minAPQt, b.maxAPQt}});
      registry.add("hV0Psi", "hV0Psi", HistType::kTH2F, {{b.nBinsV0Psi, b.minPt, TMath::PiOver2()}, {b.nBinsV0Psi, b.minPt, b.maxV0Psi}});
      if (selectCascades) {
        registry.add("hCascPt", "pT", HistType::kTH1F, {{b.nBinsCascPt, b.minPt, b.maxCascPt}});
        registry.add("hCascEtaPhi", "#eta vs. #varphi", HistType::kTH2F, {{b.nBinsPhi, b.minPhi, b.maxPhi}, {b.nBinsEta, b.minEta, b.maxEta}});
        registry.add("hCascDCAxyPosToPV", "hCascDCAxyPosToPV", HistType::kTH1F, {{b.nBinsDCA, b.minDCA, b.maxDCA}});
        registry.add("hCascDCAxyNegToPV", "hCascDCAxyNegToPV", HistType::kTH1F, {{b.nBinsDCA, b.minDCA, b.maxDCA}});
        registry.add("hCascDCAxyBachToPV", "hCascDCAxyBachToPV", HistType::kTH1F, {{b.nBinsDCA, b.minDCA, b.maxDCA}});
        registry.add("hCascDCAzPosToPV", "hCascDCAzPosToPV", HistType::kTH1F, {{b.nBinsDCA, b.minDCA, b.maxDCA}});
        registry.add("hCascDCAzNegToPV", "hCascDCAzNegToPV", HistType::kTH1F, {{b.nBinsDCA, b.minDCA, b.maxDCA}});
        registry.add("hCascDCAzBachToPV", "hCascDCAzBachToPV", HistType::kTH1F, {{b.nBinsDCA, b.minDCA, b.maxDCA}});
        registry.add("hCascAPplot", "hCascAPplot", HistType::kTH2F, {{b.nBinsAPAlpha, b.minAPAlpha, b.maxAPAlpha}, {b.nBinsCascAPQt, b.minAPQt, b.maxCascAPQt}});
        registry.add("hCascV0APplot", "hCascV0APplot", HistType::kTH2F, {{b.nBinsAPAlpha, b.minAPAlpha, b.maxAPAlpha}, {b.nBinsAPQt, b.minAPQt, b.maxAPQt}});
        registry.add("hCascAPplotSelected", "hCascAPplotSelected", HistType::kTH2F, {{b.nBinsAPAlpha, b.minAPAlpha, b.maxAPAlpha}, {b.nBinsCascAPQt, b.minAPQt, b.maxCascAPQt}});
        registry.add("hCascV0APplotSelected", "hCascV0APplotSelected", HistType::kTH2F, {{b.nBinsAPAlpha, b.minAPAlpha, b.maxAPAlpha}, {b.nBinsAPQt, b.minAPQt, b.maxAPQt}});
        registry.add("hCascRadius", "hCascRadius", HistType::kTH1F, {{b.nBinsCascRadius, b.minPt, b.maxCascRadius}});
        registry.add("hCascV0Radius", "hCascV0Radius", HistType::kTH1F, {{b.nBinsCascRadius, b.minPt, b.maxCascRadius}});
        registry.add("hCascCosPA", "hCascCosPA", HistType::kTH1F, {{b.nBinsV0CosPA, b.minV0CosPA, b.maxV0CosPA}});
        registry.add("hCascV0CosPA", "hCascV0CosPA", HistType::kTH1F, {{b.nBinsV0CosPA, b.minV0CosPA, b.maxV0CosPA}});
        registry.add("hMassOmega", "hMassOmega", HistType::kTH2F, {{b.nBinsRadius, b.minRadius, b.maxRadius}, {b.nBinsMassOmega, b.minMassOmega, b.maxMassOmega}});
        registry.add("hMassAntiOmega", "hMassAntiOmega", HistType::kTH2F, {{b.nBinsRadius, b.minRadius, b.maxRadius}, {b.nBinsMassOmega, b.minMassOmega, b.maxMassOmega}});
        registry.add("hCascDCADau", "hCascDCADau", HistType::kTH1F, {{b.nBinsDCAV0Dau, b.minPt, b.maxDCAV0Dau}});
        registry.add("hCascV0DCADau", "hCascV0DCADau", HistType::kTH1F, {{b.nBinsDCAV0Dau, b.minPt, b.maxDCAV0Dau}});
      }
    }

    if (selectCascades == false && produceCascID == true) {
      LOGP(error, "produceCascID is available only when selectCascades is enabled");
    }
    if (cutAPOmegaUp1 < cutAPOmegaDown1) {
      LOGP(error, "cutAPOmegaUp1 must be greater than cutAPOmegaDown1");
    }
  }

  void process(aod::V0Datas const& V0s, aod::CascDatas const& Cascs, FullTracksExt const& tracks, aod::Collisions const&)
  {
    std::vector<uint8_t> pidmap;
    pidmap.clear();
    pidmap.resize(tracks.size(), 0);

    std::vector<int8_t> v0pidmap;
    v0pidmap.clear();
    if (produceV0ID.value) {
      v0pidmap.resize(V0s.size(), -1);
    }
    std::vector<int8_t> cascpidmap;
    cascpidmap.clear();
    if (produceCascID.value) {
      cascpidmap.resize(Cascs.size(), kUndef);
    }
    for (auto& V0 : V0s) {
      // if (!(V0.posTrack_as<FullTracksExt>().trackType() & o2::aod::track::TPCrefit)) {
      //   continue;
      // }
      // if (!(V0.negTrack_as<FullTracksExt>().trackType() & o2::aod::track::TPCrefit)) {
      //   continue;
      // }

      // printf("V0.collisionId = %d , collision.globalIndex = %d\n",V0.collisionId(),collision.globalIndex());
      if (fillhisto) {
        registry.fill(HIST("hV0Candidate"), 1);
      }

      const auto& posTrack = V0.posTrack_as<FullTracksExt>();
      const auto& negTrack = V0.negTrack_as<FullTracksExt>();

      bool isRejectV0{false};
      for (const auto& prong : {posTrack, negTrack}) {
        isRejectV0 = isRejectV0 || std::fabs(prong.eta()) > 0.9;
        isRejectV0 = isRejectV0 || prong.tpcNClsCrossedRows() < mincrossedrows;
        isRejectV0 = isRejectV0 || prong.tpcChi2NCl() > maxchi2tpc;
        isRejectV0 = isRejectV0 || std::fabs(prong.dcaXY()) < dcamin;
        isRejectV0 = isRejectV0 || std::fabs(prong.dcaXY()) > dcamax;
      }
      isRejectV0 = isRejectV0 || (posTrack.sign() * negTrack.sign() > 0);

      if (isRejectV0)
        continue;

      float V0dca = V0.dcaV0daughters();
      float V0CosinePA = V0.v0cosPA();
      float V0radius = V0.v0radius();

      if (V0dca > dcav0dau) {
        continue;
      }

      if (V0CosinePA < v0cospa) {
        continue;
      }

      if (V0radius < v0Rmin || v0Rmax < V0radius) {
        continue;
      }
      if (fillhisto) {
        registry.fill(HIST("hV0Candidate"), 2);
      }
      float mGamma = V0.mGamma();
      float mK0S = V0.mK0Short();
      float mLambda = V0.mLambda();
      float mAntiLambda = V0.mAntiLambda();
      float psipair = V0.psipair();

      if (fillhisto) {
        registry.fill(HIST("hV0Pt"), V0.pt());
        registry.fill(HIST("hV0EtaPhi"), V0.phi(), V0.eta());
        registry.fill(HIST("hDCAxyPosToPV"), V0.posTrack_as<FullTracksExt>().dcaXY());
        registry.fill(HIST("hDCAxyNegToPV"), V0.negTrack_as<FullTracksExt>().dcaXY());
        registry.fill(HIST("hDCAzPosToPV"), V0.posTrack_as<FullTracksExt>().dcaZ());
        registry.fill(HIST("hDCAzNegToPV"), V0.negTrack_as<FullTracksExt>().dcaZ());
        registry.fill(HIST("hV0APplot"), V0.alpha(), V0.qtarm());
        registry.fill(HIST("hV0Radius"), V0radius);
        registry.fill(HIST("hV0CosPA"), V0CosinePA);
        registry.fill(HIST("hDCAV0Dau"), V0dca);
      }

      int v0id = checkV0(V0.alpha(), V0.qtarm());
      if (v0id < 0) {
        // printf("This is not [Gamma/K0S/Lambda/AntiLambda] candidate.\n");
        continue;
      }
      if (fillhisto) {
        registry.fill(HIST("hV0APplotSelected"), V0.alpha(), V0.qtarm());
      }

      auto storeV0AddID = [&](auto gix, auto id) {
        if (produceV0ID.value) {
          v0pidmap[gix] = id;
        }
      };
      if (v0id == kGamma) { // photon conversion
        if (fillhisto) {
          registry.fill(HIST("hMassGamma"), V0radius, mGamma);
          registry.fill(HIST("hV0Psi"), psipair, mGamma);
        }
        if (mGamma < v0max_mee && std::abs(V0.posTrack_as<FullTracksExt>().tpcNSigmaEl()) < cutNsigmaElTPC && std::abs(V0.negTrack_as<FullTracksExt>().tpcNSigmaEl()) < cutNsigmaElTPC && std::abs(psipair) < maxpsipair) {
          pidmap[V0.posTrackId()] |= (uint8_t(1) << kGamma);
          pidmap[V0.negTrackId()] |= (uint8_t(1) << kGamma);
          storeV0AddID(V0.globalIndex(), kGamma);
          if (fillhisto) {
            registry.fill(HIST("hGammaRxy"), V0.x(), V0.y());
          }
        }
      } else if (v0id == kK0S) { // K0S-> pi pi
        if (fillhisto) {
          registry.fill(HIST("hMassK0S"), V0radius, mK0S);
          registry.fill(HIST("hMassK0SPt"), V0.pt(), mK0S);
          registry.fill(HIST("hMassK0SEta"), V0.eta(), mK0S);
          registry.fill(HIST("hMassK0SPhi"), V0.phi(), mK0S);
        }
        if ((0.48 < mK0S && mK0S < 0.51) && std::abs(V0.posTrack_as<FullTracksExt>().tpcNSigmaPi()) < cutNsigmaPiTPC && std::abs(V0.negTrack_as<FullTracksExt>().tpcNSigmaPi()) < cutNsigmaPiTPC) {
          pidmap[V0.posTrackId()] |= (uint8_t(1) << kK0S);
          pidmap[V0.negTrackId()] |= (uint8_t(1) << kK0S);
          storeV0AddID(V0.globalIndex(), kK0S);
        }
      } else if (v0id == kLambda) { // L->p + pi-
        if (fillhisto) {
          registry.fill(HIST("hMassLambda"), V0radius, mLambda);
        }
        if (v0id == kLambda && (1.110 < mLambda && mLambda < 1.120) && std::abs(V0.posTrack_as<FullTracksExt>().tpcNSigmaPr()) < cutNsigmaPrTPC && std::abs(V0.negTrack_as<FullTracksExt>().tpcNSigmaPi()) < cutNsigmaPiTPC) {
          pidmap[V0.posTrackId()] |= (uint8_t(1) << kLambda);
          pidmap[V0.negTrackId()] |= (uint8_t(1) << kLambda);
          storeV0AddID(V0.globalIndex(), kLambda);
        }
      } else if (v0id == kAntiLambda) { // Lbar -> pbar + pi+
        if (fillhisto) {
          registry.fill(HIST("hMassAntiLambda"), V0radius, mAntiLambda);
        }
        if ((1.110 < mAntiLambda && mAntiLambda < 1.120) && std::abs(V0.posTrack_as<FullTracksExt>().tpcNSigmaPi()) < cutNsigmaPiTPC && std::abs(V0.negTrack_as<FullTracksExt>().tpcNSigmaPr()) < cutNsigmaPrTPC) {
          pidmap[V0.posTrackId()] |= (uint8_t(1) << kAntiLambda);
          pidmap[V0.negTrackId()] |= (uint8_t(1) << kAntiLambda);
          storeV0AddID(V0.globalIndex(), kAntiLambda);
        }
      }
      // printf("posTrackId = %d\n",V0.posTrackId());
      // printf("negTrackId = %d\n",V0.negTrackId());

    } // end of V0 loop
    if (produceV0ID.value) {
      for (auto& V0 : V0s) {
        v0mapID(v0pidmap[V0.globalIndex()]);
      }
    }

    if (selectCascades) {
      for (const auto& casc : Cascs) {
        if (fillhisto) {
          registry.fill(HIST("hCascCandidate"), 1);
        }

        const auto& posTrack = casc.posTrack_as<FullTracksExt>();
        const auto& negTrack = casc.negTrack_as<FullTracksExt>();
        const auto& bachelor = casc.bachelor_as<FullTracksExt>();

        bool isRejectCascade{false};
        for (const auto& prong : {posTrack, negTrack, bachelor}) {
          isRejectCascade = isRejectCascade || std::fabs(prong.eta()) > 0.9;
          isRejectCascade = isRejectCascade || prong.tpcNClsCrossedRows() < mincrossedrows;
          isRejectCascade = isRejectCascade || prong.tpcChi2NCl() > maxchi2tpc;
          isRejectCascade = isRejectCascade || std::fabs(prong.dcaXY()) < dcamin;
          isRejectCascade = isRejectCascade || std::fabs(prong.dcaXY()) > dcamax;
        }
        isRejectCascade = isRejectCascade || (posTrack.sign() * negTrack.sign() > 0);

        if (isRejectCascade)
          continue;

        if (fillhisto) {
          registry.fill(HIST("hCascCandidate"), 2);
        }

        auto collision = casc.collision_as<aod::Collisions>();
        const float collisionX = collision.posX();
        const float collisionY = collision.posY();
        const float collisionZ = collision.posZ();

        const float cascDca = casc.dcacascdaughters(); // NOTE the name of getter is misleading. In case of no-KF this is sqrt(Chi2)
        const float cascV0Dca = casc.dcaV0daughters(); // NOTE the name of getter is misleading. In case of kfDoDCAFitterPreMinimV0 this is sqrt(Chi2)
        const float cascRadius = casc.cascradius();
        const float cascV0Radius = casc.dcaV0daughters();
        const float cascCosinePA = casc.casccosPA(collisionX, collisionY, collisionZ);
        const float cascV0CosinePA = casc.v0cosPA(collisionX, collisionY, collisionZ);

        if (cascDca > cascDcaMax) {
          continue;
        }
        if (cascV0Dca > cascV0DcaMax) {
          continue;
        }
        if (cascRadius < cascRadiusMin || cascRadius > cascRadiusMax) {
          continue;
        }
        if (cascV0Radius < cascV0RadiusMin || cascV0Radius > cascV0RadiusMax) {
          continue;
        }
        if (cascCosinePA < cascCosinePAMin) {
          continue;
        }
        if (cascV0CosinePA < cascV0CosinePAMin || cascV0CosinePA > cascV0CosinePAMax) {
          continue;
        }

        const float mOmega = casc.mOmega();
        const float mV0Lambda = casc.mLambda();
        const float alpha = casc.alpha();
        const float qt = casc.qtarm();
        const float v0Alpha = casc.v0Alpha();
        const float v0Qt = casc.v0Qtarm();

        if (fillhisto) {
          registry.fill(HIST("hCascPt"), casc.pt());
          registry.fill(HIST("hCascEtaPhi"), casc.phi(), casc.eta());
          registry.fill(HIST("hCascDCAxyPosToPV"), casc.posTrack_as<FullTracksExt>().dcaXY());
          registry.fill(HIST("hCascDCAxyNegToPV"), casc.negTrack_as<FullTracksExt>().dcaXY());
          registry.fill(HIST("hCascDCAxyBachToPV"), casc.bachelor_as<FullTracksExt>().dcaXY());
          registry.fill(HIST("hCascDCAzPosToPV"), casc.posTrack_as<FullTracksExt>().dcaZ());
          registry.fill(HIST("hCascDCAzNegToPV"), casc.negTrack_as<FullTracksExt>().dcaZ());
          registry.fill(HIST("hCascDCAzBachToPV"), casc.bachelor_as<FullTracksExt>().dcaZ());
          registry.fill(HIST("hCascAPplot"), alpha, qt);
          registry.fill(HIST("hCascV0APplot"), v0Alpha, v0Qt);
          registry.fill(HIST("hCascRadius"), cascRadius);
          registry.fill(HIST("hCascV0Radius"), cascV0Radius);
          registry.fill(HIST("hCascCosPA"), cascCosinePA);
          registry.fill(HIST("hCascV0CosPA"), cascV0CosinePA);
          registry.fill(HIST("hCascDCADau"), cascDca);
          registry.fill(HIST("hCascV0DCADau"), cascV0Dca);
        }

        const int cascid = checkCascade(alpha, qt);
        const int v0id = checkV0(v0Alpha, v0Qt);
        if (cascid < 0) {
          continue;
        }
        if (v0id != kLambda && v0id != kAntiLambda) {
          continue;
        }
        if (fillhisto) {
          registry.fill(HIST("hCascAPplotSelected"), alpha, qt);
          registry.fill(HIST("hCascV0APplotSelected"), v0Alpha, v0Qt);
        }

        auto storeCascAddID = [&](auto gix, auto id) {
          if (produceCascID.value) {
            cascpidmap[gix] = id;
          }
        };

        if (cascid == kOmega && v0id == kLambda) {
          if (fillhisto) {
            registry.fill(HIST("hMassOmega"), cascRadius, mOmega);
          }
          if (cutMassOmegaLow < mOmega && mOmega < cutMassOmegaHigh && cutMassCascV0Low < mV0Lambda && mV0Lambda < cutMassCascV0High && std::abs(casc.posTrack_as<FullTracksExt>().tpcNSigmaPr()) < cutNsigmaPrTPC && std::abs(casc.negTrack_as<FullTracksExt>().tpcNSigmaPi()) < cutNsigmaPiTPC && std::abs(casc.bachelor_as<FullTracksExt>().tpcNSigmaKa()) < cutNsigmaKaTPC) {
            pidmap[casc.bachelorId()] |= (uint8_t(1) << kOmega);
            storeCascAddID(casc.globalIndex(), kOmega);
          }
        } else if (cascid == kAntiOmega && v0id == kAntiLambda) {
          if (fillhisto) {
            registry.fill(HIST("hMassAntiOmega"), cascRadius, mOmega);
          }
          if (cutMassOmegaLow < mOmega && mOmega < cutMassOmegaHigh && cutMassCascV0Low < mV0Lambda && mV0Lambda < cutMassCascV0High && std::abs(casc.posTrack_as<FullTracksExt>().tpcNSigmaPi()) < cutNsigmaPiTPC && std::abs(casc.negTrack_as<FullTracksExt>().tpcNSigmaPr()) < cutNsigmaPrTPC && std::abs(casc.bachelor_as<FullTracksExt>().tpcNSigmaKa()) < cutNsigmaKaTPC) {
            pidmap[casc.bachelorId()] |= (uint8_t(1) << kAntiOmega);
            storeCascAddID(casc.globalIndex(), kAntiOmega);
          }
        }
      } // end of Casc loop
      if (produceCascID.value) {
        for (auto& casc : Cascs) {
          cascmapID(cascpidmap[casc.globalIndex()]);
        }
      }
    }
    for (auto& track : tracks) {
      // printf("setting pidmap[%lld] = %d\n",track.globalIndex(),pidmap[track.globalIndex()]);
      v0bits(pidmap[track.globalIndex()]);
    } // end of track loop

  } // end of process
};

struct trackPIDQA {
  Configurable<float> dcamin{"dcamin", 0.0, "dcamin"};
  Configurable<float> dcamax{"dcamax", 1e+10, "dcamax"};
  Configurable<int> mincrossedrows{"mincrossedrows", 70, "min crossed rows"};
  Configurable<float> maxchi2tpc{"maxchi2tpc", 4.0, "max chi2/NclsTPC"};
  Configurable<bool> fillDQHisto{"fillDQHisto", false, "flag to fill dq histograms"};
  Configurable<std::string> fConfigAddTrackHistogram{"cfgAddTrackHistogram", "", "Comma separated list of dq histograms"};

  // Histogram binning configurables for trackPIDQA
  // Event counter histogram
  Configurable<int> nBinsEventCounter{"nBinsEventCounter", 5, "Number of bins for event counter"};
  Configurable<float> minEventCounter{"minEventCounter", 0.5f, "Minimum value for event counter"};
  Configurable<float> maxEventCounter{"maxEventCounter", 5.5f, "Maximum value for event counter"};

  // Track pT histograms
  Configurable<int> nBinsTrackPt{"nBinsTrackPt", 100, "Number of bins for track pT"};
  Configurable<float> minTrackPt{"minTrackPt", 0.1f, "Minimum track pT"};
  Configurable<float> maxTrackPt{"maxTrackPt", 10.0f, "Maximum track pT"};

  // Track eta-phi histograms
  Configurable<int> nBinsTrackPhi{"nBinsTrackPhi", 63, "Number of bins for track phi"};
  Configurable<float> minTrackPhi{"minTrackPhi", 0.0f, "Minimum track phi"};
  Configurable<float> maxTrackPhi{"maxTrackPhi", 6.3f, "Maximum track phi"};

  Configurable<int> nBinsTrackEta{"nBinsTrackEta", 20, "Number of bins for track eta"};
  Configurable<float> minTrackEta{"minTrackEta", -1.0f, "Minimum track eta"};
  Configurable<float> maxTrackEta{"maxTrackEta", 1.0f, "Maximum track eta"};

  // TPC dE/dx histograms
  Configurable<int> nBinsTPCPin{"nBinsTPCPin", 200, "Number of bins for TPC momentum"};
  Configurable<float> minTPCPin{"minTPCPin", 0.1f, "Minimum TPC momentum"};
  Configurable<float> maxTPCPin{"maxTPCPin", 10.0f, "Maximum TPC momentum"};

  Configurable<int> nBinsTPCdEdx{"nBinsTPCdEdx", 200, "Number of bins for TPC dE/dx"};
  Configurable<float> minTPCdEdx{"minTPCdEdx", 0.0f, "Minimum TPC dE/dx"};
  Configurable<float> maxTPCdEdx{"maxTPCdEdx", 200.0f, "Maximum TPC dE/dx"};

  // TPC nSigma histograms
  Configurable<int> nBinsTPCnSigma{"nBinsTPCnSigma", 200, "Number of bins for TPC nSigma"};
  Configurable<float> minTPCnSigma{"minTPCnSigma", -10.0f, "Minimum TPC nSigma"};
  Configurable<float> maxTPCnSigma{"maxTPCnSigma", 10.0f, "Maximum TPC nSigma"};

  // TOF beta histograms
  Configurable<int> nBinsTOFbeta{"nBinsTOFbeta", 120, "Number of bins for TOF beta"};
  Configurable<float> minTOFbeta{"minTOFbeta", 0.0f, "Minimum TOF beta"};
  Configurable<float> maxTOFbeta{"maxTOFbeta", 1.2f, "Maximum TOF beta"};

  // TOF nSigma histograms
  Configurable<int> nBinsTOFnSigma{"nBinsTOFnSigma", 200, "Number of bins for TOF nSigma"};
  Configurable<float> minTOFnSigma{"minTOFnSigma", -10.0f, "Minimum TOF nSigma"};
  Configurable<float> maxTOFnSigma{"maxTOFnSigma", 10.0f, "Maximum TOF nSigma"};

  // Log binning configurables
  Configurable<int> logAxis{"logAxis", 1, "Alias: if true, enables logarithmic binning for momentum axes (same as useLogBinningMomentum)"};

  HistogramRegistry registry{"registry"};
  OutputObj<THashList> fOutputList{"output"}; //! the histogram manager output list
  HistogramManager* fHistMan;

  void init(o2::framework::InitContext& context)
  {
    bool enableBarrelHistos = context.mOptions.get<bool>("processQA");
    if (enableBarrelHistos) {
      registry.add("hEventCounter", "hEventCounter", HistType::kTH1F, {{nBinsEventCounter, minEventCounter, maxEventCounter}});

      // Create momentum axis based on log binning setting
      AxisSpec momentumAxis{nBinsTPCPin, minTPCPin, maxTPCPin, "p (GeV/c)"};
      AxisSpec transverseMomentumAxis{nBinsTrackPt, minTrackPt, maxTrackPt, "p_{T} (GeV/c)"};
      if (logAxis) {
        if (minTPCPin <= 0.0f || minTrackPt <= 0.0f) {
          minTPCPin = 0.1f;
          minTrackPt = 0.1f;
          LOGP(warn, "minTPCPin and minTrackPt must be positive for logarithmic binning. Setting to 0.1");
        }
        momentumAxis = AxisSpec{nBinsTPCPin, minTPCPin, maxTPCPin, "p (GeV/c)"};
        transverseMomentumAxis = AxisSpec{nBinsTrackPt, minTrackPt, maxTrackPt, "p_{T} (GeV/c)"};
        momentumAxis.makeLogarithmic();
        transverseMomentumAxis.makeLogarithmic();
      }

      const AxisSpec dedxAxis{nBinsTPCdEdx, minTPCdEdx, maxTPCdEdx, "TPC dE/dx (a.u.)"};
      const AxisSpec nSigmaAxis{nBinsTPCnSigma, minTPCnSigma, maxTPCnSigma, "n#sigma"};
      const AxisSpec betaAxis{nBinsTOFbeta, minTOFbeta, maxTOFbeta, "#beta"};
      const AxisSpec tofnSigmaAxis{nBinsTOFnSigma, minTOFnSigma, maxTOFnSigma, "n#sigma"};

      registry.add("hTrackPt_all", "pT", HistType::kTH1F, {transverseMomentumAxis});
      registry.add("hTrackEtaPhi_all", "#eta vs. #varphi", HistType::kTH2F, {{nBinsTrackPhi, minTrackPhi, maxTrackPhi}, {nBinsTrackEta, minTrackEta, maxTrackEta}});
      registry.add("h2TPCdEdx_Pin_all", "TPC dEdx vs. p_{in}", HistType::kTH2F, {momentumAxis, dedxAxis});
      registry.add("h2TOFbeta_Pin_all", "TOF #beta vs. p_{in}", HistType::kTH2F, {momentumAxis, betaAxis});

      registry.add("hTrackPt", "pT", HistType::kTH1F, {transverseMomentumAxis});
      registry.add("hTrackEtaPhi", "#eta vs. #varphi", HistType::kTH2F, {{nBinsTrackPhi, minTrackPhi, maxTrackPhi}, {nBinsTrackEta, minTrackEta, maxTrackEta}});

      registry.add("h2TPCdEdx_Pin", "TPC dEdx vs. p_{in}", HistType::kTH2F, {momentumAxis, dedxAxis});
      registry.add("h2TPCdEdx_Pin_El", "TPC dEdx vs. p_{in}", HistType::kTH2F, {momentumAxis, dedxAxis});
      registry.add("h2TPCdEdx_Pin_Pi", "TPC dEdx vs. p_{in}", HistType::kTH2F, {momentumAxis, dedxAxis});
      registry.add("h2TPCdEdx_Pin_Ka", "TPC dEdx vs. p_{in}", HistType::kTH2F, {momentumAxis, dedxAxis});
      registry.add("h2TPCdEdx_Pin_Pr", "TPC dEdx vs. p_{in}", HistType::kTH2F, {momentumAxis, dedxAxis});

      registry.add("h2TPCnSigma_Pin_El", "TPC n#sigma_{e} vs. p_{in}", HistType::kTH2F, {momentumAxis, nSigmaAxis});
      registry.add("h2TPCnSigma_Pin_Pi", "TPC n#sigma_{#pi} vs. p_{in}", HistType::kTH2F, {momentumAxis, nSigmaAxis});
      registry.add("h2TPCnSigma_Pin_Ka", "TPC n#sigma_{K} vs. p_{in}", HistType::kTH2F, {momentumAxis, nSigmaAxis});
      registry.add("h2TPCnSigma_Pin_Pr", "TPC n#sigma_{p} vs. p_{in}", HistType::kTH2F, {momentumAxis, nSigmaAxis});

      registry.add("h2TOFbeta_Pin", "TOF #beta vs. p_{in}", HistType::kTH2F, {momentumAxis, betaAxis});
      registry.add("h2TOFbeta_Pin_El", "TOF #beta vs. p_{in}", HistType::kTH2F, {momentumAxis, betaAxis});
      registry.add("h2TOFbeta_Pin_Pi", "TOF #beta vs. p_{in}", HistType::kTH2F, {momentumAxis, betaAxis});
      registry.add("h2TOFbeta_Pin_Ka", "TOF #beta vs. p_{in}", HistType::kTH2F, {momentumAxis, betaAxis});
      registry.add("h2TOFbeta_Pin_Pr", "TOF #beta vs. p_{in}", HistType::kTH2F, {momentumAxis, betaAxis});

      registry.add("h2TOFnSigma_Pin_El", "TOF n#sigma_{e} vs. p_{in}", HistType::kTH2F, {momentumAxis, tofnSigmaAxis});
      registry.add("h2TOFnSigma_Pin_Pi", "TOF n#sigma_{#pi} vs. p_{in}", HistType::kTH2F, {momentumAxis, tofnSigmaAxis});
      registry.add("h2TOFnSigma_Pin_Ka", "TOF n#sigma_{K} vs. p_{in}", HistType::kTH2F, {momentumAxis, tofnSigmaAxis});
      registry.add("h2TOFnSigma_Pin_Pr", "TOF n#sigma_{p} vs. p_{in}", HistType::kTH2F, {momentumAxis, tofnSigmaAxis});
    }

    if (fillDQHisto) {
      TString histClasses = "";
      VarManager::SetDefaultVarNames();
      fHistMan = new HistogramManager("analysisHistos", "aa", VarManager::kNVars);
      fHistMan->SetUseDefaultVariableNames(kTRUE);
      fHistMan->SetDefaultVarNames(VarManager::fgVariableNames, VarManager::fgVariableUnits);
      histClasses += "V0Track_all;";
      histClasses += "V0Track_electron;";
      histClasses += "V0Track_pion;";
      histClasses += "V0Track_proton;";
      DefineHistograms(histClasses);
      VarManager::SetUseVars(fHistMan->GetUsedVars()); // provide the list of required variables so that VarManager knows what to fill
      fOutputList.setObject(fHistMan->GetMainHistogramList());
    }
  }

  void processQA(soa::Join<aod::Collisions, aod::EvSels>::iterator const& collision,
                 soa::Join<FullTracksExt, aod::pidTOFFullEl, aod::pidTOFFullPi, aod::pidTOFFullKa, aod::pidTOFFullPr, aod::pidTOFbeta, aod::V0Bits> const& tracks)
  {
    registry.fill(HIST("hEventCounter"), 1.0); // all

    if (!collision.sel8()) {
      return;
    }
    registry.fill(HIST("hEventCounter"), 2.0); // FT0VX i.e. FT0and

    if (collision.numContrib() < 0.5) {
      return;
    }
    registry.fill(HIST("hEventCounter"), 3.0); // Ncontrib > 0

    if (abs(collision.posZ()) > 10.0) {
      return;
    }
    registry.fill(HIST("hEventCounter"), 4.0); //|Zvtx| < 10 cm

    for (auto& track : tracks) {
      if (!track.has_collision()) {
        continue;
      }

      if (track.tpcNClsCrossedRows() < mincrossedrows) {
        continue;
      }

      if (track.tpcChi2NCl() > maxchi2tpc) {
        continue;
      }

      if (std::fabs(track.dcaXY()) < dcamin) {
        continue;
      }

      if (std::fabs(track.dcaXY()) > dcamax) {
        continue;
      }

      if (std::fabs(track.eta()) > 0.9) {
        continue;
      }

      registry.fill(HIST("hTrackPt_all"), track.pt());
      registry.fill(HIST("hTrackEtaPhi_all"), track.phi(), track.eta());
      registry.fill(HIST("h2TPCdEdx_Pin_all"), track.tpcInnerParam(), track.tpcSignal());
      registry.fill(HIST("h2TOFbeta_Pin_all"), track.tpcInnerParam(), track.beta());
      if (track.pidbit() > 0) {
        registry.fill(HIST("hTrackPt"), track.pt());
        registry.fill(HIST("hTrackEtaPhi"), track.phi(), track.eta());
        registry.fill(HIST("h2TPCdEdx_Pin"), track.tpcInnerParam(), track.tpcSignal());
        registry.fill(HIST("h2TOFbeta_Pin"), track.tpcInnerParam(), track.beta());
      }

      if (static_cast<bool>(track.pidbit() & (1 << v0selector::kGamma))) {
        registry.fill(HIST("h2TPCdEdx_Pin_El"), track.tpcInnerParam(), track.tpcSignal());
        registry.fill(HIST("h2TOFbeta_Pin_El"), track.tpcInnerParam(), track.beta());
        registry.fill(HIST("h2TPCnSigma_Pin_El"), track.tpcInnerParam(), track.tpcNSigmaEl());
        registry.fill(HIST("h2TOFnSigma_Pin_El"), track.tpcInnerParam(), track.tofNSigmaEl());
      }
      if (static_cast<bool>(track.pidbit() & (1 << v0selector::kK0S))) {
        registry.fill(HIST("h2TPCdEdx_Pin_Pi"), track.tpcInnerParam(), track.tpcSignal());
        registry.fill(HIST("h2TOFbeta_Pin_Pi"), track.tpcInnerParam(), track.beta());
        registry.fill(HIST("h2TPCnSigma_Pin_Pi"), track.tpcInnerParam(), track.tpcNSigmaPi());
        registry.fill(HIST("h2TOFnSigma_Pin_Pi"), track.tpcInnerParam(), track.tofNSigmaPi());
      }
      if (static_cast<bool>(track.pidbit() & (1 << v0selector::kLambda))) {
        if (track.sign() > 0) {
          registry.fill(HIST("h2TPCdEdx_Pin_Pr"), track.tpcInnerParam(), track.tpcSignal());
          registry.fill(HIST("h2TOFbeta_Pin_Pr"), track.tpcInnerParam(), track.beta());
          registry.fill(HIST("h2TPCnSigma_Pin_Pr"), track.tpcInnerParam(), track.tpcNSigmaPr());
          registry.fill(HIST("h2TOFnSigma_Pin_Pr"), track.tpcInnerParam(), track.tofNSigmaPr());
        } else {
          registry.fill(HIST("h2TPCdEdx_Pin_Pi"), track.tpcInnerParam(), track.tpcSignal());
          registry.fill(HIST("h2TOFbeta_Pin_Pi"), track.tpcInnerParam(), track.beta());
          registry.fill(HIST("h2TPCnSigma_Pin_Pi"), track.tpcInnerParam(), track.tpcNSigmaPi());
          registry.fill(HIST("h2TOFnSigma_Pin_Pi"), track.tpcInnerParam(), track.tofNSigmaPi());
        }
      }
      if (static_cast<bool>(track.pidbit() & (1 << v0selector::kAntiLambda))) {
        if (track.sign() > 0) {
          registry.fill(HIST("h2TPCdEdx_Pin_Pi"), track.tpcInnerParam(), track.tpcSignal());
          registry.fill(HIST("h2TOFbeta_Pin_Pi"), track.tpcInnerParam(), track.beta());
          registry.fill(HIST("h2TPCnSigma_Pin_Pi"), track.tpcInnerParam(), track.tpcNSigmaPi());
          registry.fill(HIST("h2TOFnSigma_Pin_Pi"), track.tpcInnerParam(), track.tofNSigmaPi());
        } else {
          registry.fill(HIST("h2TPCdEdx_Pin_Pr"), track.tpcInnerParam(), track.tpcSignal());
          registry.fill(HIST("h2TOFbeta_Pin_Pr"), track.tpcInnerParam(), track.beta());
          registry.fill(HIST("h2TPCnSigma_Pin_Pr"), track.tpcInnerParam(), track.tpcNSigmaPr());
          registry.fill(HIST("h2TOFnSigma_Pin_Pr"), track.tpcInnerParam(), track.tofNSigmaPr());
        }
      }
      if (static_cast<bool>(track.pidbit() & (1 << v0selector::kOmega))) {
        registry.fill(HIST("h2TPCdEdx_Pin_Ka"), track.tpcInnerParam(), track.tpcSignal());
        registry.fill(HIST("h2TOFbeta_Pin_Ka"), track.tpcInnerParam(), track.beta());
        registry.fill(HIST("h2TPCnSigma_Pin_Ka"), track.tpcInnerParam(), track.tpcNSigmaKa());
        registry.fill(HIST("h2TOFnSigma_Pin_Ka"), track.tpcInnerParam(), track.tofNSigmaKa());
      }

      if (fillDQHisto) {
        VarManager::FillTrack<gkTrackFillMap>(track);
        fHistMan->FillHistClass("V0Track_all", VarManager::fgValues);
        if (static_cast<bool>(track.pidbit() & (1 << v0selector::kGamma))) {
          fHistMan->FillHistClass("V0Track_electron", VarManager::fgValues);
        }
        if (static_cast<bool>(track.pidbit() & (1 << v0selector::kK0S))) {
          fHistMan->FillHistClass("V0Track_pion", VarManager::fgValues);
        }
        if (static_cast<bool>(track.pidbit() & (1 << v0selector::kLambda))) {
          if (track.sign() > 0) {
            fHistMan->FillHistClass("V0Track_proton", VarManager::fgValues);
          } else {
            fHistMan->FillHistClass("V0Track_pion", VarManager::fgValues);
          }
        }
        if (static_cast<bool>(track.pidbit() & (1 << v0selector::kAntiLambda))) {
          if (track.sign() > 0) {
            fHistMan->FillHistClass("V0Track_pion", VarManager::fgValues);
          } else {
            fHistMan->FillHistClass("V0Track_proton", VarManager::fgValues);
          }
        }
      }

    } // end of track loop
  } // end of process

  void DefineHistograms(TString histClasses)
  {
    std::unique_ptr<TObjArray> objArray(histClasses.Tokenize(";"));
    for (Int_t iclass = 0; iclass < objArray->GetEntries(); ++iclass) {
      TString classStr = objArray->At(iclass)->GetName();
      fHistMan->AddHistClass(classStr.Data());

      // fill the THn histograms
      if (classStr.Contains("V0Track_electron")) {
        o2::aod::dqhistograms::DefineHistograms(fHistMan, objArray->At(iclass)->GetName(), "track", "postcalib_electron");
      }
      if (classStr.Contains("V0Track_pion")) {
        o2::aod::dqhistograms::DefineHistograms(fHistMan, objArray->At(iclass)->GetName(), "track", "postcalib_pion");
      }
      if (classStr.Contains("V0Track_proton")) {
        o2::aod::dqhistograms::DefineHistograms(fHistMan, objArray->At(iclass)->GetName(), "track", "postcalib_proton");
      }

      TString histTrackName = fConfigAddTrackHistogram.value;
      if (classStr.Contains("Track")) {
        o2::aod::dqhistograms::DefineHistograms(fHistMan, objArray->At(iclass)->GetName(), "track", histTrackName);
      }
    }
  }

  void processDummy(soa::Join<aod::Collisions, aod::EvSels>::iterator const&)
  {
    // do nothing
  }

  PROCESS_SWITCH(trackPIDQA, processQA, "Run PID QA for barrel tracks", true);
  PROCESS_SWITCH(trackPIDQA, processDummy, "Dummy function", false);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<v0selector>(cfgc, TaskName{"v0-selector"}),
    adaptAnalysisTask<trackPIDQA>(cfgc, TaskName{"track-pid-qa"})};
}
