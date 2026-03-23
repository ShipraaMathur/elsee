import { View, Text, StyleSheet, TouchableOpacity, Image, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useAuth } from '../context/AuthContext';

export default function LoginScreen() {
  const { login, isLoading } = useAuth();

  return (
    <SafeAreaView style={s.container}>
      <View style={s.inner}>
        <View style={s.logoWrap}>
          <Text style={s.logoIcon}>👁</Text>
          <Text style={s.logoText}>SeeForMe</Text>
          <Text style={s.tagline}>AI vision assistant for everyone</Text>
        </View>

        <View style={s.features}>
          {['Real-time obstacle detection', 'Ask questions about your surroundings', 'Session history & replay'].map((f) => (
            <View key={f} style={s.featureRow}>
              <Text style={s.featureDot}>✦</Text>
              <Text style={s.featureText}>{f}</Text>
            </View>
          ))}
        </View>

        <TouchableOpacity style={s.loginBtn} onPress={login} disabled={isLoading}>
          {isLoading
            ? <ActivityIndicator color="#080B12" />
            : <Text style={s.loginBtnText}>Sign in with Auth0</Text>}
        </TouchableOpacity>

        <Text style={s.terms}>By signing in you agree to our Terms & Privacy Policy</Text>
      </View>
    </SafeAreaView>
  );
}

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#080B12' },
  inner: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 32, gap: 32 },
  logoWrap: { alignItems: 'center', gap: 8 },
  logoIcon: { fontSize: 64 },
  logoText: { fontSize: 36, fontWeight: '900', color: '#00F5C4', letterSpacing: 2 },
  tagline: { color: '#5A6580', fontSize: 14 },
  features: { width: '100%', gap: 14 },
  featureRow: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  featureDot: { color: '#00F5C4', fontSize: 12 },
  featureText: { color: '#C8CDD8', fontSize: 15 },
  loginBtn: {
    width: '100%', backgroundColor: '#00F5C4',
    borderRadius: 14, padding: 16, alignItems: 'center',
  },
  loginBtnText: { color: '#080B12', fontWeight: '800', fontSize: 16 },
  terms: { color: '#3A4260', fontSize: 11, textAlign: 'center' },
});