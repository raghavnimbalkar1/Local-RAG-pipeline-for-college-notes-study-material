// src/App.js
import { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [currentlyTyping, setCurrentlyTyping] = useState(false);
  const [typingContent, setTypingContent] = useState("");
  const [typingIndex, setTypingIndex] = useState(0);
  const bottomRef = useRef();

  // Function to simulate typing effect
  useEffect(() => {
    if (currentlyTyping && typingContent && typingIndex < typingContent.length) {
      const timer = setTimeout(() => {
        setTypingIndex(prev => prev + 1);
      }, 1); 
      
      return () => clearTimeout(timer);
    } else if (typingIndex >= typingContent.length && currentlyTyping) {
      // Finished typing
      setCurrentlyTyping(false);
      setMessages(prev => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1] = {
          ...newMessages[newMessages.length - 1],
          content: typingContent
        };
        return newMessages;
      });
      setTypingContent("");
      setTypingIndex(0);
    }
  }, [currentlyTyping, typingContent, typingIndex]);

  const askQuestion = async () => {
    if (!question.trim()) return;
    const userMessage = { role: "user", content: question };
    setMessages((prev) => [...prev, userMessage]);
    setQuestion("");
    setLoading(true);

    try {
      const res = await axios.post("http://localhost:8000/query", { question });
      
      // Add a placeholder message that will be replaced with typing animation
      const placeholderMessage = { role: "assistant", content: "" };
      setMessages((prev) => [...prev, placeholderMessage]);
      
      // Start the typing animation
      setTypingContent(res.data.answer);
      setCurrentlyTyping(true);
      setTypingIndex(0);
      
    } catch (err) {
      const botMessage = { role: "assistant", content: "Error: " + err.message };
      setMessages((prev) => [...prev, botMessage]);
    }

    setLoading(false);
  };

  // Scroll to bottom on new message or while typing
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, typingIndex]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  };

  // Clear conversation
  const clearChat = () => {
    setMessages([]);
    setCurrentlyTyping(false);
    setTypingContent("");
    setTypingIndex(0);
  };

  return (
    <div className="App">
      <div className="sidebar">
        <div className="sidebar-header">
          <button className="new-chat-btn" onClick={clearChat}>
            <span className="plus-icon">+</span> New Chat
          </button>
        </div>
        <div className="chat-history">
          {/* You could implement chat history here */}
        </div>
        <div className="sidebar-footer">
          <div className="user-profile">
            <div className="avatar">U</div>
            <span>User</span>
          </div>
        </div>
      </div>
      
      <div className="main-content">
        <div className="chat-header">
          <h1>StudyBuddy</h1>
          {messages.length > 0 && (
            <button className="clear-btn" onClick={clearChat}>
              Clear Conversation
            </button>
          )}
        </div>
        
        <div className="chat-container">
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <div className="welcome-icon">🤖</div>
              <h2>How can I help you today?</h2>
              <p>Ask anything about your college notes and materials</p>
              <div className="suggestion-chips">
                <div className="chip">Explain machine learning</div>
                <div className="chip">Summarize this chapter</div>
                <div className="chip">Help with math problem</div>
                <div className="chip">Define key terms</div>
              </div>
            </div>
          ) : (
            <div className="message-container">
              {messages.map((msg, i) => (
                <div key={i} className={`message ${msg.role}`}>
                  <div className="avatar">
                    {msg.role === "user" ? "U" : "AI"}
                  </div>
                  <div className="message-content">
                    {/* Show typing animation for the last message if it's currently being typed */}
                    {i === messages.length - 1 && currentlyTyping ? (
                      <>
                        {typingContent.substring(0, typingIndex)}
                        <span className="typing-cursor">|</span>
                      </>
                    ) : (
                      msg.content.split('\n').map((line, index) => (
                        <p key={index}>{line}</p>
                      ))
                    )}
                  </div>
                </div>
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </div>
        
        <div className="input-container">
          <div className="input-box">
            <textarea
              rows={1}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Message StudyBuddy..."
              onKeyDown={handleKeyDown}
              disabled={currentlyTyping}
            />
            <button 
              onClick={askQuestion} 
              disabled={loading || !question.trim() || currentlyTyping}
              className="send-button"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path>
              </svg>
            </button>
          </div>
          <div className="disclaimer">
            StudyBuddy can make mistakes. Consider checking important information.
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;