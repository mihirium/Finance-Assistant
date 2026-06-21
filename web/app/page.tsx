"use client";

import {
  ArrowUp,
  Bot,
  Clock3,
  Database,
  FileText,
  LineChart,
  Newspaper,
  Search,
  User
} from "lucide-react";
import { FormEvent, useMemo, useRef, useState } from "react";

type Source = {
  title: string;
  url: string;
  sourceType: "news";
  score?: number;
  excerpt?: string;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

const starterQuestions = [
  "What happened in markets today?",
  "What are the biggest company stories today?",
  "What themes are driving stocks today?"
];

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Ask what happened in markets today. I will retrieve current financial news and answer with citations."
    }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const latestSources = useMemo(() => {
    return [...messages].reverse().find((message) => message.sources?.length)?.sources ?? [];
  }, [messages]);

  async function submitQuestion(question: string) {
    const trimmed = question.trim();
    if (!trimmed || isLoading) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed
    };
    setMessages((current) => [...current, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed })
      });
      if (!response.ok) {
        throw new Error(`Chat request failed: ${response.status}`);
      }
      const data = (await response.json()) as { answer: string; sources?: Source[] };
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: data.answer,
          sources: data.sources ?? []
        }
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content:
            error instanceof Error
              ? `I could not reach the chat backend yet. ${error.message}`
              : "I could not reach the chat backend yet."
        }
      ]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submitQuestion(input);
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <LineChart size={20} />
          </div>
          <div>
            <h1>Finance AI</h1>
            <p>Daily financial news</p>
          </div>
        </div>

        <section className="panel">
          <div className="panel-title">
            <Database size={16} />
            <span>Data Status</span>
          </div>
          <div className="status-row">
            <span>pgvector</span>
            <strong>Connected soon</strong>
          </div>
          <div className="status-row">
            <span>Embeddings</span>
            <strong>Hugging Face</strong>
          </div>
          <div className="status-row">
            <span>Content</span>
            <strong>Today&apos;s news</strong>
          </div>
        </section>

        <section className="panel sources-panel">
          <div className="panel-title">
            <FileText size={16} />
            <span>Latest Sources</span>
          </div>
          {latestSources.length === 0 ? (
            <p className="muted">Citations from retrieved news will appear here.</p>
          ) : (
            <div className="source-list">
              {latestSources.map((source, index) => (
                <a key={`${source.url}-${index}`} href={source.url} target="_blank" rel="noreferrer" className="source-item">
                  <span className="source-index">[{index + 1}]</span>
                  <span>
                    <strong>{source.title}</strong>
                    <small>
                      {source.sourceType}
                      {typeof source.score === "number" ? ` · ${source.score.toFixed(2)}` : ""}
                    </small>
                  </span>
                </a>
              ))}
            </div>
          )}
        </section>
      </aside>

      <section className="chat-workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Local RAG Workspace</p>
            <h2>Ask grounded finance questions</h2>
          </div>
          <div className="topbar-meta">
            <span>
              <Newspaper size={15} />
              News
            </span>
            <span>
              <Clock3 size={15} />
              Updated daily
            </span>
          </div>
        </header>

        <div className="messages">
          {messages.map((message) => (
            <article key={message.id} className={`message ${message.role}`}>
              <div className="avatar" aria-hidden="true">
                {message.role === "assistant" ? <Bot size={18} /> : <User size={18} />}
              </div>
              <div className="message-body">
                <p>{message.content}</p>
                {message.sources && message.sources.length > 0 ? (
                  <div className="citation-strip">
                    {message.sources.slice(0, 4).map((source, index) => (
                      <a key={`${source.url}-${index}`} href={source.url} target="_blank" rel="noreferrer">
                        [{index + 1}] {source.sourceType}
                      </a>
                    ))}
                  </div>
                ) : null}
              </div>
            </article>
          ))}
          {isLoading ? (
            <article className="message assistant">
              <div className="avatar" aria-hidden="true">
                <Bot size={18} />
              </div>
              <div className="message-body loading-text">Retrieving today&apos;s news and drafting answer...</div>
            </article>
          ) : null}
        </div>

        <div className="starter-row">
          {starterQuestions.map((question) => (
            <button key={question} type="button" onClick={() => void submitQuestion(question)}>
              <Search size={15} />
              <span>{question}</span>
            </button>
          ))}
        </div>

        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask what happened in markets today..."
            rows={2}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void submitQuestion(input);
              }
            }}
          />
          <button type="submit" disabled={!input.trim() || isLoading} aria-label="Send message">
            <ArrowUp size={20} />
          </button>
        </form>
      </section>
    </main>
  );
}
