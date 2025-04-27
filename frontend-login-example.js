// Example frontend implementation of login with JSON API

import axios from 'axios';

// Configure axios defaults
const API_BASE_URL = 'http://localhost:8000/api/v1';
axios.defaults.baseURL = API_BASE_URL;

// Check if we have a token and set it in axios headers
const token = localStorage.getItem('token');
if (token) {
  axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
}

// Login function using the JSON API
export const login = async (email, password) => {
  try {
    // Use JSON payload instead of form data
    const response = await axios.post('/auth/login', {
      email,
      password
    });
    
    // Store token in localStorage
    const { access_token } = response.data;
    localStorage.setItem('token', access_token);
    
    // Set authorization header for future requests
    axios.defaults.headers.common['Authorization'] = `Bearer ${access_token}`;
    
    // Return the user data (optional - you might want to fetch user details)
    return getUserProfile();
  } catch (error) {
    // Handle different types of errors
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      throw new Error(error.response.data.detail || 'Login failed');
    } else if (error.request) {
      // The request was made but no response was received
      throw new Error('No response from server');
    } else {
      // Something happened in setting up the request that triggered an Error
      throw new Error('Error setting up request');
    }
  }
};

// Logout function
export const logout = () => {
  // Remove token from localStorage
  localStorage.removeItem('token');
  
  // Remove authorization header
  delete axios.defaults.headers.common['Authorization'];
};

// Get user profile
export const getUserProfile = async () => {
  try {
    const response = await axios.get('/auth/me');
    return response.data;
  } catch (error) {
    if (error.response && error.response.status === 401) {
      // If unauthorized, clear token and redirect to login
      logout();
      throw new Error('Session expired. Please log in again.');
    }
    throw error;
  }
};

// Example React component for login form
/*
import React, { useState } from 'react';
import { login } from './auth-service';

const LoginForm = ({ onLoginSuccess }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    
    try {
      const userData = await login(email, password);
      setIsLoading(false);
      onLoginSuccess(userData);
    } catch (error) {
      setIsLoading(false);
      setError(error.message);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      {error && <div className="error">{error}</div>}
      <div>
        <label htmlFor="email">Email</label>
        <input
          id="email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
      </div>
      <div>
        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
      </div>
      <button type="submit" disabled={isLoading}>
        {isLoading ? 'Logging in...' : 'Login'}
      </button>
    </form>
  );
};

export default LoginForm;
*/ 