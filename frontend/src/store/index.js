/**
 * Global app state store — HAR Deep Vision
 * Uses React context + useReducer for lightweight state management.
 * Manages: model health, inference history, and UI preferences.
 */

import React, { createContext, useContext, useReducer } from 'react';

const initialState = {
  health: { status: 'connecting', model_state: null, device: 'unknown' },
  modelInfo: null,
  metrics: null,
  inferenceHistory: [],   // last N predictions [{activity, confidence, ts}]
  preferences: {
    usePose: true,
    maxHistoryLength: 20,
  },
};

function reducer(state, action) {
  switch (action.type) {
    case 'SET_HEALTH':
      return { ...state, health: action.payload };
    case 'SET_MODEL_INFO':
      return { ...state, modelInfo: action.payload };
    case 'SET_METRICS':
      return { ...state, metrics: action.payload };
    case 'ADD_INFERENCE': {
      const history = [
        { ...action.payload, ts: Date.now() },
        ...state.inferenceHistory,
      ].slice(0, state.preferences.maxHistoryLength);
      return { ...state, inferenceHistory: history };
    }
    case 'SET_USE_POSE':
      return { ...state, preferences: { ...state.preferences, usePose: action.payload } };
    case 'CLEAR_HISTORY':
      return { ...state, inferenceHistory: [] };
    default:
      return state;
  }
}

const StoreContext = createContext(null);

export function StoreProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <StoreContext.Provider value={{ state, dispatch }}>
      {children}
    </StoreContext.Provider>
  );
}

export function useStore() {
  const ctx = useContext(StoreContext);
  if (!ctx) throw new Error('useStore must be used within a StoreProvider');
  return ctx;
}

export default StoreProvider;
