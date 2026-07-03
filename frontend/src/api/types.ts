/**
 * API response contract — mirror of the backend Pydantic models in
 * `backend/app/models/*.py`. The deep-learning module is the return-based GRU
 * model from ADR-0006 (no LSTM/VMD, no MAPE/R²): metrics are RMSE, MAE and
 * directional accuracy, and the model output is a 10-day return.
 */

export type SentimentLabel = 'positivo' | 'negativo' | 'neutral'
export type Trend = 'alcista' | 'bajista' | 'neutral'

export type NewsItem = {
  title: string
  url: string
  source: string
}

export type SentimentResult = {
  label: SentimentLabel
  /** Normalized sentiment in [-1, 1]. */
  score: number
  /** Model confidence in [0, 1]. */
  confidence: number
  explanation: string
  influential_news: NewsItem[]
}

export type ModelMetrics = {
  rmse: number
  mae: number
  /** Sign-agreement rate in [0, 1]. */
  directional_accuracy: number
}

export type DLResult = {
  trend: Trend
  /** Model output: cumulative 10-day return, in percent. */
  predicted_return_pct: number
  /** Derived: current_price * (1 + predicted_return_pct / 100). */
  predicted_price: number
  current_price: number
  horizon_days: number
  trained_at: string
  metrics: ModelMetrics
}

export type FundamentalResult = {
  /** Composite score in [0, 10]. */
  score: number
  metrics: Record<string, number | string>
  llm_analysis: string
  cached_at: string
}

/** 0–10 score of each technical block (null if the block could not be computed). */
export type TechnicalBlockScores = {
  momentum: number | null
  trend: number | null
  risk_stability: number | null
  confirmation: number | null
}

export type TechnicalResult = {
  /** Composite score in [0, 10]. */
  score: number
  signal: Trend
  block_scores: TechnicalBlockScores
  indicators: Record<string, number | string>
  llm_analysis: string
  calculated_at: string
}

export type ReportResponse = {
  ticker: string
  generated_at: string
  sentiment: SentimentResult | null
  deep_learning: DLResult | null
  fundamental: FundamentalResult | null
  technical: TechnicalResult | null
  global_conclusion: string
  disclaimer: string
  /** True when no GRU model exists for the ticker (RF-27). */
  partial_support: boolean
}
