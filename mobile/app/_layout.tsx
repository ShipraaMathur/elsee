import { Tabs, useRouter, useSegments } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { useEffect } from 'react';
import { StyleSheet, Text } from 'react-native';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import { RemoteConfigProvider } from '../context/RemoteConfigContext';
import { AuthProvider, useAuth } from '../context/AuthContext';

function TabIcon({ emoji, focused }: { emoji: string; focused: boolean }) {
  return <Text style={{ fontSize: 20, opacity: focused ? 1 : 0.4 }}>{emoji}</Text>;
}

// This inner component can use the auth hook because it's inside AuthProvider
function RootLayoutNav() {
  const { user, isLoading } = useAuth();
  const segments = useSegments();
  const router = useRouter();

  // useEffect(() => {
  //   if (isLoading) return;
  //   const inAuthGroup = segments[0] === 'login';
  //   if (!user && !inAuthGroup) {
  //     router.replace('/login');
  //   } else if (user && inAuthGroup) {
  //     router.replace('/');
  //   }
  // }, [user, isLoading, segments]);
  useEffect(() => {
    if (isLoading) return; // still restoring session, wait

    const inLoginPage = segments[0] === 'login';

    if (!user && !inLoginPage) {
      router.replace('/login');
    } else if (user && inLoginPage) {
      router.replace('/ask'); // go to Ask tab, not just '/'
    }
  }, [user, isLoading]); // remove segments from deps — causes loop

  return (
    <Tabs
      initialRouteName="ask"
      screenOptions={{
        headerShown: false,
        tabBarStyle: styles.tabBar,
        tabBarActiveTintColor: '#00F5C4',
        tabBarInactiveTintColor: '#444',
        tabBarLabelStyle: styles.tabLabel,
      }}
    >
      <Tabs.Screen name="index" options={{ title: 'Live', tabBarIcon: ({ focused }) => <TabIcon emoji="👁" focused={focused} /> }} />
      <Tabs.Screen name="ask" options={{ title: 'Ask', tabBarIcon: ({ focused }) => <TabIcon emoji="🎙" focused={focused} /> }} />
      <Tabs.Screen name="history" options={{ title: 'History', tabBarIcon: ({ focused }) => <TabIcon emoji="📋" focused={focused} /> }} />
      <Tabs.Screen name="settings" options={{ title: 'Settings', tabBarIcon: ({ focused }) => <TabIcon emoji="⚙️" focused={focused} /> }} />
      {/* Hide login from tab bar */}
      <Tabs.Screen name="login" options={{ href: null }} />
    </Tabs>
  );
}

export default function RootLayout() {
  return (
    <AuthProvider>
      <RemoteConfigProvider>
        <GestureHandlerRootView style={{ flex: 1 }}>
          <SafeAreaProvider>
            <StatusBar style="light" />
            <RootLayoutNav />
          </SafeAreaProvider>
        </GestureHandlerRootView>
      </RemoteConfigProvider>
    </AuthProvider>
  );
}

const styles = StyleSheet.create({
  tabBar: {
    backgroundColor: '#0E1320',
    borderTopColor: '#1E2740',
    borderTopWidth: 1,
    paddingBottom: 8,
    paddingTop: 8,
    height: 70,
  },
  tabLabel: { fontSize: 11, fontWeight: '600', marginTop: 2 },
});