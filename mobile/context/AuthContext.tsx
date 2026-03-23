import React, { createContext, useContext, useState, useEffect } from 'react';
import * as AuthSession from 'expo-auth-session';
import * as WebBrowser from 'expo-web-browser';
import * as Crypto from 'expo-crypto';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Constants from 'expo-constants';

WebBrowser.maybeCompleteAuthSession();

const AUTH0_DOMAIN = process.env.EXPO_PUBLIC_AUTH0_DOMAIN || '';
const AUTH0_CLIENT_ID = process.env.EXPO_PUBLIC_AUTH0_CLIENT_ID || '';

type AuthUser = {
  sub: string;
  email: string;
  name: string;
  picture?: string;
  accessToken: string;
};

type AuthContextType = {
  user: AuthUser | null;
  isLoading: boolean;
  login: () => Promise<void>;
  logout: () => Promise<void>;
};

const AuthContext = createContext<AuthContextType>({
  user: null,
  isLoading: true,
  login: async () => {},
  logout: async () => {},
});

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // const redirectUri = AuthSession.makeRedirectUri({ scheme: 'seefore', path: 'auth' });
  const redirectUri = AuthSession.makeRedirectUri({
    scheme: 'seefore',
    path: 'auth',
  });

  const discovery = AuthSession.useAutoDiscovery(`https://${AUTH0_DOMAIN}`);

  // Restore session on app launch
  useEffect(() => {
    AsyncStorage.getItem('sf_user').then((raw) => {
      if (raw) setUser(JSON.parse(raw));
      setIsLoading(false);
    });
  }, []);

  const login = async () => {
    if (!discovery) return;
    try {
      const state = await Crypto.digestStringAsync(
        Crypto.CryptoDigestAlgorithm.SHA256,
        Math.random().toString()
      );

      // const request = new AuthSession.AuthRequest({
      //   clientId: AUTH0_CLIENT_ID,
      //   redirectUri,
      //   scopes: ['openid', 'profile', 'email'],
      //   extraParams: { audience: process.env.EXPO_PUBLIC_AUTH0_AUDIENCE || '' },
      // });
      const request = new AuthSession.AuthRequest({
        clientId: AUTH0_CLIENT_ID,
        redirectUri,
        scopes: ['openid', 'profile', 'email'],
        extraParams: { 
          audience: process.env.EXPO_PUBLIC_AUTH0_AUDIENCE || 'https://api.seefore.tech'
        },
      });

      console.log('Redirect URI:', redirectUri);
      const result = await request.promptAsync(discovery);

      if (result.type !== 'success') return;

      // Exchange code for tokens
      const tokenRes = await AuthSession.exchangeCodeAsync(
        {
          clientId: AUTH0_CLIENT_ID,
          code: result.params.code,
          redirectUri,
          extraParams: { code_verifier: request.codeVerifier || '' },
        },
        discovery
      );

      // Fetch user info
      const userInfoRes = await fetch(`https://${AUTH0_DOMAIN}/userinfo`, {
        headers: { Authorization: `Bearer ${tokenRes.accessToken}` },
      });
      const userInfo = await userInfoRes.json();

      const authUser: AuthUser = {
        sub: userInfo.sub,
        email: userInfo.email,
        name: userInfo.name,
        picture: userInfo.picture,
        accessToken: tokenRes.accessToken,
      };

      await AsyncStorage.setItem('sf_user', JSON.stringify(authUser));
      setUser(authUser);
    } catch (e) {
      console.error('Login error:', e);
    }
  };

  const logout = async () => {
    await AsyncStorage.removeItem('sf_user');
    setUser(null);
    // Optionally hit Auth0 logout endpoint
    await WebBrowser.openBrowserAsync(
      `https://${AUTH0_DOMAIN}/v2/logout?client_id=${AUTH0_CLIENT_ID}&returnTo=${redirectUri}`
    );
  };

  return (
    <AuthContext.Provider value={{ user, isLoading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);