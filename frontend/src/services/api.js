import axios from 'axios';

const API_BASE = '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
});

export const getHealth = async () => {
  const res = await api.get('/health');
  return res.data;
};

export const getModelInfo = async () => {
  const res = await api.get('/model-info');
  return res.data;
};

export const getClasses = async () => {
  const res = await api.get('/classes');
  return res.data;
};

export const getMetrics = async () => {
  const res = await api.get('/metrics');
  return res.data;
};

export const getInferenceStats = async () => {
  const res = await api.get('/inference-stats');
  return res.data;
};

export const getTrainingHistory = async () => {
  const res = await api.get('/results/training-history');
  return res.data;
};

export const predictVideo = async (file, usePose = true) => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await api.post(`/predict/video?use_pose=${usePose}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export const predictFrameSequence = async (files, usePose = true) => {
  const formData = new FormData();
  files.forEach((f) => formData.append('files', f));
  const res = await api.post(`/predict/frame-sequence?use_pose=${usePose}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
};

export const reloadModel = async () => {
  const res = await api.post('/model/load');
  return res.data;
};

export default api;
