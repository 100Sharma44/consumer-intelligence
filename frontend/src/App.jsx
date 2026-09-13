import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [userId, setUserId] = useState("15");
  const [consumer, setConsumer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyzeConsumer = async () => {
    if (!userId) return;

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/predict/${userId}`);

      if (!response.ok) {
        throw new Error("Consumer not found");
      }

      const data = await response.json();
      setConsumer(data);
    } catch (err) {
      setConsumer(null);
      setError("Unable to load consumer. Check the user ID and API server.");
    } finally {
      setLoading(false);
    }
  };

  const formatPercent = (value) =>
    `${(Number(value) * 100).toFixed(1)}%`;

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <div className="eyebrow">CONSUMER INTELLIGENCE</div>
          <h1>Behavioral Intelligence Dashboard</h1>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          API Connected
        </div>
      </header>

      <main className="container">
        <section className="hero">
          <div>
            <p className="label">CONSUMER ANALYSIS</p>
            <h2>Understand the next customer action.</h2>
            <p className="subtitle">
              Combine behavioral signals and ML purchase propensity
              to create actionable advertising decisions.
            </p>
          </div>

          <div className="search-box">
            <label htmlFor="userId">Consumer ID</label>

            <div className="search-row">
              <input
                id="userId"
                type="number"
                min="1"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder="Enter user ID"
              />

              <button onClick={analyzeConsumer} disabled={loading}>
                {loading ? "Analyzing..." : "Analyze"}
              </button>
            </div>
          </div>
        </section>

        {error && <div className="error">{error}</div>}

        {!consumer && !loading && !error && (
          <div className="empty-state">
            <h3>Ready to analyze a consumer</h3>
            <p>
              Enter a consumer ID above to view behavioral intelligence
              and the recommended advertising action.
            </p>
          </div>
        )}

        {consumer && (
          <>
            <section className="section">
              <div className="section-heading">
                <div>
                  <p className="label">CONSUMER OVERVIEW</p>
                  <h3>Consumer #{consumer.user_id}</h3>
                </div>
              </div>

              <div className="metrics-grid">
                <MetricCard
                  title="Purchase Propensity"
                  value={formatPercent(
                    consumer.purchase_intent_probability
                  )}
                  description="ML-estimated seven-day purchase propensity"
                  highlight
                />

                <MetricCard
                  title="Current Intent"
                  value={consumer.intent_level}
                  description="Based on recent behavioral activity"
                />

                <MetricCard
                  title="Primary Interest"
                  value={consumer.primary_interest}
                  description={`Interest score: ${consumer.interest_score}`}
                />

                <MetricCard
                  title="Price Sensitivity"
                  value={consumer.price_sensitivity_level}
                  description={`Behavioral score: ${consumer.price_sensitivity_score}`}
                />
              </div>
            </section>

            <section className="two-column">
              <div className="panel">
                <p className="label">CONSUMER SEGMENT</p>

                <div className="segment">
                  {consumer.consumer_segment.replaceAll("_", " ")}
                </div>

                <p className="panel-description">
                  This segment combines current behavioral intent,
                  purchase propensity, and observed price interaction
                  patterns.
                </p>
              </div>

              <div className="panel advertising-panel">
                <p className="label">ADVERTISING DECISION</p>

                <div className="decision">
                  <span className="decision-icon">→</span>

                  <div>
                    <div className="decision-title">
                      {consumer.recommended_action.replaceAll(
                        "_",
                        " "
                      )}
                    </div>

                    <div className="decision-subtitle">
                      Recommended advertising objective
                    </div>
                  </div>
                </div>

                <div className="recommendation-grid">
                  <div>
                    <span>Category</span>
                    <strong>{consumer.recommended_category}</strong>
                  </div>

                  <div>
                    <span>Messaging</span>
                    <strong>
                      {consumer.messaging_strategy.replaceAll(
                        "_",
                        " "
                      )}
                    </strong>
                  </div>
                </div>
              </div>
            </section>

            <section className="panel reason-panel">
              <p className="label">DECISION REASON</p>

              <p className="reason">
                {consumer.decision_reason}
              </p>
            </section>
          </>
        )}
      </main>

      <footer>
        Consumer Intelligence V1 · Behavioral signals + ML propensity
      </footer>
    </div>
  );
}

function MetricCard({
  title,
  value,
  description,
  highlight = false,
}) {
  return (
    <div className={`metric-card ${highlight ? "highlight" : ""}`}>
      <span className="metric-title">{title}</span>

      <strong className="metric-value">{value}</strong>

      <span className="metric-description">
        {description}
      </span>
    </div>
  );
}

export default App;