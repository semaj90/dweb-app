<!-- Criminal Profile Component for Legal AI App -->
<script lang="ts">
  import { User, Calendar, MapPin, AlertTriangle, Shield, Eye, FileText, Fingerprint, Camera } from 'lucide-svelte';
  import { cn } from '$lib/utils';
  import { $props, $derived } from 'svelte';

  export interface CriminalRecord {
    id: string;
    offense: string;
    date: Date;
    jurisdiction: string;
    disposition: 'convicted' | 'acquitted' | 'dismissed' | 'pending' | 'sealed';
    sentence?: string;
    caseNumber?: string;
  }

  export interface BiometricData {
    fingerprints?: string[];
    dnaProfile?: string;
    facialRecognition?: string;
    voicePrint?: string;
  }

  export interface CriminalProfile {
    id: string;
    personalInfo: {
      firstName: string;
      lastName: string;
      aliases?: string[];
      dateOfBirth: Date;
      placeOfBirth?: string;
      gender: 'male' | 'female' | 'other' | 'unknown';
      height?: string;
      weight?: string;
      eyeColor?: string;
      hairColor?: string;
      distinguishingMarks?: string[];
    };
    identification: {
      ssn?: string;
      driverLicense?: string;
      passport?: string;
      mugshots?: string[];
      biometrics?: BiometricData;
    };
    address: {
      current?: string;
      previous?: string[];
      knownAssociates?: string[];
    };
    criminalHistory: CriminalRecord[];
    riskAssessment: {
      riskLevel: 'low' | 'medium' | 'high' | 'extreme';
      flightRisk: boolean;
      violentHistory: boolean;
      reoffenseRisk: number; // 0-100
      lastUpdated: Date;
    };
    currentStatus: 'at_large' | 'incarcerated' | 'on_parole' | 'probation' | 'deceased' | 'cleared';
    warrants?: Array<{
      id: string;
      type: string;
      issueDate: Date;
      jurisdiction: string;
      status: 'active' | 'served' | 'recalled';
    }>;
    notes?: string;
  }

  export interface CriminalProfileProps {
    profile: CriminalProfile;
    viewMode?: 'full' | 'summary' | 'identification';
    showSensitiveInfo?: boolean;
    interactive?: boolean;
    onViewFullRecord?: (recordId: string) => void;
    onUpdateProfile?: (profile: CriminalProfile) => void;
    onViewMugshot?: (mugshotUrl: string) => void;
    class?: string;
  }

  let {
    profile,
    viewMode = 'full',
    showSensitiveInfo = false,
    interactive = true,
    onViewFullRecord,
    onUpdateProfile,
    onViewMugshot,
    class: className = ''
  }: CriminalProfileProps = $props();

  // Risk level configurations
  const riskConfig = {
    low: { label: 'Low Risk', class: 'bg-green-500/20 text-green-400 border-green-500/30' },
    medium: { label: 'Medium Risk', class: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30' },
    high: { label: 'High Risk', class: 'bg-orange-500/20 text-orange-400 border-orange-500/30' },
    extreme: { label: 'Extreme Risk', class: 'bg-red-500/20 text-red-400 border-red-500/30' }
  };

  // Status configurations
  const statusConfig = {
    at_large: { label: 'At Large', class: 'bg-red-500/20 text-red-400', icon: AlertTriangle },
    incarcerated: { label: 'Incarcerated', class: 'bg-gray-500/20 text-gray-400', icon: Shield },
    on_parole: { label: 'On Parole', class: 'bg-yellow-500/20 text-yellow-400', icon: Eye },
    probation: { label: 'Probation', class: 'bg-blue-500/20 text-blue-400', icon: FileText },
    deceased: { label: 'Deceased', class: 'bg-gray-500/20 text-gray-400', icon: User },
    cleared: { label: 'Cleared', class: 'bg-green-500/20 text-green-400', icon: Shield }
  };

  // Disposition configurations
  const dispositionConfig = {
    convicted: { label: 'Convicted', class: 'bg-red-500/20 text-red-400' },
    acquitted: { label: 'Acquitted', class: 'bg-green-500/20 text-green-400' },
    dismissed: { label: 'Dismissed', class: 'bg-blue-500/20 text-blue-400' },
    pending: { label: 'Pending', class: 'bg-yellow-500/20 text-yellow-400' },
    sealed: { label: 'Sealed', class: 'bg-gray-500/20 text-gray-400' }
  };

  // Calculate age
  const age = $derived(() => {
    const today = new Date();
    const birthDate = profile.personalInfo.dateOfBirth;
    let age = today.getFullYear() - birthDate.getFullYear();
    const monthDiff = today.getMonth() - birthDate.getMonth();
    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
      age--;
    }
    return age;
  });

  // Active warrants
  const activeWarrants = $derived(() => {
    return profile.warrants?.filter(warrant => warrant.status === 'active') || [];
  });

  // Recent criminal activity
  const recentRecords = $derived(() => {
    return profile.criminalHistory
      .sort((a, b) => b.date.getTime() - a.date.getTime())
      .slice(0, 5);
  });

  function formatDate(date: Date): string {
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  }

  function maskSSN(ssn: string): string {
    if (!showSensitiveInfo) {
      return `***-**-${ssn.slice(-4)}`;
    }
    return ssn;
  }

  function getFullName(): string {
    const { firstName, lastName } = profile.personalInfo;
    return `${firstName} ${lastName}`;
  }
</script>

<div class={cn(
  'criminal-profile bg-yorha-bg-secondary border border-yorha-border rounded-lg overflow-hidden',
  profile.currentStatus === 'at_large' && 'border-red-500/30',
  className
)}>
  <!-- Profile Header -->
  <div class={cn(
    'p-4 border-b border-yorha-border',
    profile.currentStatus === 'at_large' && 'bg-red-500/5'
  )}>
    <div class="flex items-start gap-4">
      <!-- Profile Photo/Mugshot -->
      <div class="shrink-0">
        {#if profile.identification.mugshots?.length}
          <button
            onclick={() => onViewMugshot?.(profile.identification.mugshots[0])}
            class="w-20 h-24 bg-yorha-bg-tertiary border border-yorha-border rounded overflow-hidden hover:border-yorha-primary/30 transition-colors group"
          >
            <div class="w-full h-full flex items-center justify-center text-yorha-text-secondary group-hover:text-yorha-primary">
              <Camera class="w-6 h-6" />
            </div>
          </button>
        {:else}
          <div class="w-20 h-24 bg-yorha-bg-tertiary border border-yorha-border rounded flex items-center justify-center">
            <User class="w-8 h-8 text-yorha-text-secondary" />
          </div>
        {/if}
      </div>

      <!-- Profile Info -->
      <div class="flex-1 min-w-0">
        <div class="flex items-start justify-between mb-2">
          <div>
            <h2 class="text-xl font-bold text-yorha-text-primary font-mono">
              {getFullName()}
            </h2>
            <div class="text-sm text-yorha-text-secondary font-mono">
              ID: {profile.id} • Age: {age}
            </div>
          </div>

          <!-- Current Status -->
          {@const statusInfo = statusConfig[profile.currentStatus]}
          {@const StatusIcon = statusInfo.icon}
          <div class="flex items-center gap-2">
            <span class={cn('px-3 py-1 text-xs font-mono rounded border', statusInfo.class)}>
              <StatusIcon class="w-3 h-3 inline mr-1" />
              {statusInfo.label}
            </span>
          </div>
        </div>

        <!-- Aliases -->
        {#if profile.personalInfo.aliases?.length}
          <div class="mb-2">
            <span class="text-xs text-yorha-text-secondary font-mono">AKA:</span>
            <span class="text-sm text-yorha-text-primary font-mono ml-2">
              {profile.personalInfo.aliases.join(', ')}
            </span>
          </div>
        {/if}

        <!-- Risk Assessment -->
        <div class="flex items-center gap-4 text-xs font-mono">
          <div class="flex items-center gap-2">
            <span class="text-yorha-text-secondary">Risk Level:</span>
            <span class={cn(
              'px-2 py-0.5 rounded border',
              riskConfig[profile.riskAssessment.riskLevel].class
            )}>
              {riskConfig[profile.riskAssessment.riskLevel].label}
            </span>
          </div>

          {#if profile.riskAssessment.flightRisk}
            <span class="px-2 py-0.5 bg-red-500/20 text-red-400 border border-red-500/30 rounded">
              FLIGHT RISK
            </span>
          {/if}

          {#if profile.riskAssessment.violentHistory}
            <span class="px-2 py-0.5 bg-red-500/20 text-red-400 border border-red-500/30 rounded">
              VIOLENT HISTORY
            </span>
          {/if}
        </div>
      </div>
    </div>

    <!-- Active Warrants Alert -->
    {#if activeWarrants.length > 0}
      <div class="mt-3 p-3 bg-red-500/10 border border-red-500/20 rounded">
        <div class="flex items-center gap-2 text-red-400 font-medium text-sm font-mono mb-1">
          <AlertTriangle class="w-4 h-4" />
          {activeWarrants.length} Active Warrant{activeWarrants.length !== 1 ? 's' : ''}
        </div>
        {#each activeWarrants as warrant}
          <div class="text-xs text-red-300 font-mono">
            {warrant.type} - {warrant.jurisdiction} ({formatDate(warrant.issueDate)})
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <!-- Profile Content -->
  <div class="p-4 space-y-4">
    <!-- Personal Information -->
    {#if viewMode === 'full' || viewMode === 'identification'}
      <div>
        <h3 class="text-sm font-semibold text-yorha-text-primary font-mono mb-3 uppercase">
          Personal Information
        </h3>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 text-sm font-mono">
          <div>
            <span class="text-yorha-text-secondary">Date of Birth:</span>
            <div class="text-yorha-text-primary">{formatDate(profile.personalInfo.dateOfBirth)}</div>
          </div>
          {#if profile.personalInfo.placeOfBirth}
            <div>
              <span class="text-yorha-text-secondary">Place of Birth:</span>
              <div class="text-yorha-text-primary">{profile.personalInfo.placeOfBirth}</div>
            </div>
          {/if}
          <div>
            <span class="text-yorha-text-secondary">Gender:</span>
            <div class="text-yorha-text-primary capitalize">{profile.personalInfo.gender}</div>
          </div>
          {#if profile.personalInfo.height}
            <div>
              <span class="text-yorha-text-secondary">Height:</span>
              <div class="text-yorha-text-primary">{profile.personalInfo.height}</div>
            </div>
          {/if}
          {#if profile.personalInfo.weight}
            <div>
              <span class="text-yorha-text-secondary">Weight:</span>
              <div class="text-yorha-text-primary">{profile.personalInfo.weight}</div>
            </div>
          {/if}
          {#if profile.personalInfo.eyeColor}
            <div>
              <span class="text-yorha-text-secondary">Eye Color:</span>
              <div class="text-yorha-text-primary">{profile.personalInfo.eyeColor}</div>
            </div>
          {/if}
          {#if profile.personalInfo.hairColor}
            <div>
              <span class="text-yorha-text-secondary">Hair Color:</span>
              <div class="text-yorha-text-primary">{profile.personalInfo.hairColor}</div>
            </div>
          {/if}
        </div>

        <!-- Distinguishing Marks -->
        {#if profile.personalInfo.distinguishingMarks?.length}
          <div class="mt-3">
            <span class="text-xs text-yorha-text-secondary font-mono">Distinguishing Marks:</span>
            <div class="flex flex-wrap gap-2 mt-1">
              {#each profile.personalInfo.distinguishingMarks as mark}
                <span class="px-2 py-1 text-xs font-mono bg-yorha-bg-tertiary text-yorha-text-primary rounded border border-yorha-border">
                  {mark}
                </span>
              {/each}
            </div>
          </div>
        {/if}
      </div>
    {/if}

    <!-- Identification -->
    {#if viewMode === 'full' || viewMode === 'identification'}
      <div>
        <h3 class="text-sm font-semibold text-yorha-text-primary font-mono mb-3 uppercase">
          Identification
        </h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm font-mono">
          {#if profile.identification.ssn}
            <div>
              <span class="text-yorha-text-secondary">SSN:</span>
              <div class="text-yorha-text-primary">{maskSSN(profile.identification.ssn)}</div>
            </div>
          {/if}
          {#if profile.identification.driverLicense}
            <div>
              <span class="text-yorha-text-secondary">Driver's License:</span>
              <div class="text-yorha-text-primary">{profile.identification.driverLicense}</div>
            </div>
          {/if}
          {#if profile.identification.passport}
            <div>
              <span class="text-yorha-text-secondary">Passport:</span>
              <div class="text-yorha-text-primary">{profile.identification.passport}</div>
            </div>
          {/if}
        </div>

        <!-- Biometric Data -->
        {#if profile.identification.biometrics}
          <div class="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3 text-xs font-mono">
            {#if profile.identification.biometrics.fingerprints?.length}
              <div class="flex items-center gap-2">
                <Fingerprint class="w-4 h-4 text-yorha-text-secondary" />
                <span class="text-yorha-text-primary">
                  {profile.identification.biometrics.fingerprints.length} fingerprint record{profile.identification.biometrics.fingerprints.length !== 1 ? 's' : ''}
                </span>
              </div>
            {/if}
            {#if profile.identification.biometrics.dnaProfile}
              <div class="flex items-center gap-2 text-yorha-text-primary">
                <Shield class="w-4 h-4 text-yorha-text-secondary" />
                DNA profile on file
              </div>
            {/if}
            {#if profile.identification.biometrics.facialRecognition}
              <div class="flex items-center gap-2 text-yorha-text-primary">
                <Camera class="w-4 h-4 text-yorha-text-secondary" />
                Facial recognition data
              </div>
            {/if}
          </div>
        {/if}
      </div>
    {/if}

    <!-- Criminal History -->
    {#if viewMode === 'full' || viewMode === 'summary'}
      <div>
        <h3 class="text-sm font-semibold text-yorha-text-primary font-mono mb-3 uppercase">
          Criminal History ({profile.criminalHistory.length} record{profile.criminalHistory.length !== 1 ? 's' : ''})
        </h3>
        
        {#if recentRecords.length === 0}
          <p class="text-sm text-yorha-text-secondary font-mono">No criminal records found</p>
        {:else}
          <div class="space-y-3">
            {#each recentRecords as record}
              <div class="bg-yorha-bg-tertiary border border-yorha-border rounded p-3">
                <div class="flex items-start justify-between mb-2">
                  <div>
                    <h4 class="text-sm font-medium text-yorha-text-primary font-mono">
                      {record.offense}
                    </h4>
                    <div class="text-xs text-yorha-text-secondary font-mono">
                      {formatDate(record.date)} • {record.jurisdiction}
                      {#if record.caseNumber}
                        • Case #{record.caseNumber}
                      {/if}
                    </div>
                  </div>
                  <span class={cn(
                    'px-2 py-1 text-xs font-mono rounded border',
                    dispositionConfig[record.disposition].class
                  )}>
                    {dispositionConfig[record.disposition].label}
                  </span>
                </div>
                
                {#if record.sentence}
                  <p class="text-xs text-yorha-text-secondary font-mono">
                    Sentence: {record.sentence}
                  </p>
                {/if}

                {#if interactive && onViewFullRecord}
                  <button
                    onclick={() => onViewFullRecord?.(record.id)}
                    class="mt-2 text-xs font-mono text-yorha-primary hover:text-yorha-accent transition-colors"
                  >
                    View Full Record
                  </button>
                {/if}
              </div>
            {/each}

            {#if profile.criminalHistory.length > 5}
              <div class="text-center">
                <span class="text-xs font-mono text-yorha-text-secondary">
                  Showing {recentRecords.length} of {profile.criminalHistory.length} records
                </span>
              </div>
            {/if}
          </div>
        {/if}
      </div>
    {/if}

    <!-- Additional Notes -->
    {#if profile.notes && viewMode === 'full'}
      <div>
        <h3 class="text-sm font-semibold text-yorha-text-primary font-mono mb-3 uppercase">
          Notes
        </h3>
        <div class="bg-yorha-bg-tertiary border border-yorha-border rounded p-3">
          <p class="text-sm text-yorha-text-primary font-mono whitespace-pre-wrap">
            {profile.notes}
          </p>
        </div>
      </div>
    {/if}
  </div>

  <!-- Footer Actions -->
  {#if interactive && onUpdateProfile}
    <div class="px-4 py-3 bg-yorha-bg-tertiary border-t border-yorha-border">
      <div class="flex justify-end">
        <button
          onclick={() => onUpdateProfile?.(profile)}
          class="px-4 py-2 text-sm font-mono bg-yorha-primary/10 text-yorha-primary border border-yorha-primary/20 rounded hover:bg-yorha-primary/20 transition-colors"
        >
          Update Profile
        </button>
      </div>
    </div>
  {/if}
</div>

<style>
  .criminal-profile {
    transition: all 0.2s ease;
  }
</style>