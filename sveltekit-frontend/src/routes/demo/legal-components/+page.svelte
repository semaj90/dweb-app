<!-- Comprehensive Legal Components Demo -->
<script lang="ts">
  import { 
    BitsDataTable, BitsCombobox, BitsDatePicker, ToastProvider,
    CaseTimeline, ChainOfCustodyTracker, LegalPrecedentCard, CriminalProfile,
    type DataTableColumn, type ComboboxOption
  } from '$lib/components/ui/bits';
  import { Calendar, FileText, Users, Scale, Shield, AlertTriangle } from 'lucide-svelte';
  import { $state } from 'svelte';

  // Page state
  let activeTab = $state('data-table');
  let toastProvider: any;

  // Sample data for components
  const evidenceData = [
    {
      id: 'EV001',
      itemNumber: 'EV-2024-001',
      description: 'Blood sample from crime scene',
      category: 'biological',
      dateCollected: '2024-01-15',
      location: 'Crime Scene A',
      custodian: 'Det. Sarah Johnson',
      status: 'analyzed'
    },
    {
      id: 'EV002', 
      itemNumber: 'EV-2024-002',
      description: 'Security footage USB drive',
      category: 'digital',
      dateCollected: '2024-01-16',
      location: 'Evidence Room',
      custodian: 'Officer Mike Chen',
      status: 'pending'
    },
    {
      id: 'EV003',
      itemNumber: 'EV-2024-003', 
      description: 'Fingerprints from door handle',
      category: 'physical',
      dateCollected: '2024-01-15',
      location: 'Crime Scene A',
      custodian: 'CSI Team Lead',
      status: 'processed'
    }
  ];

  const evidenceColumns: DataTableColumn[] = [
    { key: 'itemNumber', label: 'Item #', sortable: true, width: '120px' },
    { key: 'description', label: 'Description', sortable: true },
    { key: 'category', label: 'Category', sortable: true, width: '100px' },
    { key: 'dateCollected', label: 'Date Collected', sortable: true, width: '120px' },
    { key: 'custodian', label: 'Current Custodian', sortable: true, width: '150px' },
    { 
      key: 'status', 
      label: 'Status', 
      sortable: true, 
      width: '100px',
      render: (value) => `<span class="px-2 py-1 text-xs rounded ${
        value === 'analyzed' ? 'bg-green-500/20 text-green-400' :
        value === 'pending' ? 'bg-yellow-500/20 text-yellow-400' :
        'bg-blue-500/20 text-blue-400'
      }">${value.toUpperCase()}</span>`
    }
  ];

  // Legal entities for combobox
  const legalEntities: ComboboxOption[] = [
    { value: 'smith_corp', label: 'Smith Corporation', category: 'Corporate' },
    { value: 'johnson_llc', label: 'Johnson & Associates LLC', category: 'Corporate' },
    { value: 'john_doe', label: 'John Doe', category: 'Individual', description: 'Plaintiff' },
    { value: 'jane_smith', label: 'Jane Smith', category: 'Individual', description: 'Defendant' },
    { value: 'city_court', label: 'City Court System', category: 'Government' },
    { value: 'state_ag', label: 'State Attorney General', category: 'Government' }
  ];

  // Sample case timeline
  const caseTimelineEvents = [
    {
      id: '1',
      date: new Date('2024-01-15'),
      title: 'Case Filed',
      description: 'Initial complaint filed by plaintiff',
      type: 'filing' as const,
      status: 'completed' as const,
      participants: ['John Doe', 'Attorney Wilson']
    },
    {
      id: '2', 
      date: new Date('2024-01-22'),
      title: 'Discovery Phase Begin',
      description: 'Discovery period initiated',
      type: 'milestone' as const,
      status: 'completed' as const
    },
    {
      id: '3',
      date: new Date('2024-02-15'),
      title: 'Deposition Scheduled',
      description: 'Key witness deposition',
      type: 'hearing' as const,
      status: 'pending' as const,
      location: 'Conference Room A'
    }
  ];

  // Sample evidence item for chain of custody
  const evidenceItem = {
    id: 'EV001',
    itemNumber: 'EV-2024-001',
    description: 'Blood sample from crime scene - DNA analysis',
    category: 'biological' as const,
    collectedDate: new Date('2024-01-15'),
    collectedBy: 'CSI Team Alpha',
    currentCustodian: 'Det. Sarah Johnson',
    location: 'Evidence Room B-12',
    condition: 'excellent' as const,
    sealed: true,
    compromised: false,
    chainOfCustody: [
      {
        id: 'CT001',
        timestamp: new Date('2024-01-15T10:30:00'),
        fromPerson: 'CSI Tech Williams',
        toPerson: 'Det. Sarah Johnson',
        location: 'Crime Scene to Evidence Room',
        reason: 'Initial evidence collection transfer',
        condition: 'excellent' as const,
        signature: 'S. Johnson',
        witnessSignature: 'M. Chen',
        verified: true,
        notes: 'Sample collected using sterile procedures'
      },
      {
        id: 'CT002',
        timestamp: new Date('2024-01-16T14:15:00'),
        fromPerson: 'Det. Sarah Johnson',
        toPerson: 'Lab Tech Rodriguez',
        location: 'Evidence Room to Forensics Lab',
        reason: 'DNA analysis processing',
        condition: 'excellent' as const,
        signature: 'A. Rodriguez',
        verified: true
      }
    ]
  };

  // Sample legal precedent
  const legalPrecedent = {
    id: 'PREC001',
    caseNumber: '2022-SC-1234',
    caseName: 'State v. Anderson Digital Privacy Case',
    court: 'State Supreme Court',
    jurisdiction: 'state' as const,
    date: new Date('2022-03-15'),
    judge: 'Chief Justice Maria Rodriguez',
    summary: 'Landmark case establishing digital privacy rights in criminal investigations involving encrypted devices and cloud storage.',
    keyIssues: ['Digital Privacy', 'Fourth Amendment', 'Encrypted Devices', 'Cloud Storage'],
    holding: 'Law enforcement must obtain specific warrants for encrypted device searches',
    reasoning: [
      'Digital privacy extends traditional Fourth Amendment protections',
      'Encrypted devices require heightened warrant specificity',
      'Cloud storage maintains reasonable expectation of privacy'
    ],
    legalAreas: ['Criminal Law', 'Constitutional Law', 'Technology Law'],
    citations: 45,
    relevanceScore: 92,
    similarityScore: 78,
    precedentType: 'binding' as const,
    overruled: false
  };

  // Sample criminal profile
  const criminalProfile = {
    id: 'CRIM001',
    personalInfo: {
      firstName: 'Marcus',
      lastName: 'Williams',
      aliases: ['M. Will', 'Mac W.'],
      dateOfBirth: new Date('1985-06-20'),
      placeOfBirth: 'Chicago, IL',
      gender: 'male' as const,
      height: '6\'2"',
      weight: '180 lbs',
      eyeColor: 'Brown',
      hairColor: 'Black',
      distinguishingMarks: ['Scar on left cheek', 'Tattoo on right arm']
    },
    identification: {
      ssn: '123-45-6789',
      driverLicense: 'IL123456789',
      mugshots: ['mugshot1.jpg'],
      biometrics: {
        fingerprints: ['print1.dat', 'print2.dat'],
        dnaProfile: 'dna_profile_mw001.dat'
      }
    },
    address: {
      current: '123 Main St, Chicago, IL 60601',
      previous: ['456 Oak Ave, Chicago, IL', '789 Pine St, Chicago, IL']
    },
    criminalHistory: [
      {
        id: 'CR001',
        offense: 'Burglary in the Second Degree',
        date: new Date('2020-05-15'),
        jurisdiction: 'Cook County, IL',
        disposition: 'convicted' as const,
        sentence: '2 years probation',
        caseNumber: '2020-CR-5678'
      },
      {
        id: 'CR002',
        offense: 'Theft of Motor Vehicle',
        date: new Date('2018-11-22'),
        jurisdiction: 'Chicago PD',
        disposition: 'dismissed' as const,
        caseNumber: '2018-CR-9012'
      }
    ],
    riskAssessment: {
      riskLevel: 'medium' as const,
      flightRisk: false,
      violentHistory: false,
      reoffenseRisk: 35,
      lastUpdated: new Date('2024-01-01')
    },
    currentStatus: 'on_parole' as const,
    warrants: [
      {
        id: 'W001',
        type: 'Probation Violation',
        issueDate: new Date('2024-01-10'),
        jurisdiction: 'Cook County',
        status: 'active' as const
      }
    ]
  };

  // Event handlers
  function handleRowClick(row: any) {
    toastProvider?.showEvidenceProcessed(`Evidence ${row.itemNumber} details viewed`);
  }

  function handleExport(data: any[]) {
    toastProvider?.addToast({
      variant: 'success',
      title: 'Export Complete',
      description: `Exported ${data.length} evidence records`
    });
  }

  let selectedEntity = $state('');
  let selectedDeadline = $state<Date | undefined>(undefined);

  function handleEntityChange(value: string | undefined) {
    selectedEntity = value || '';
    if (value) {
      const entity = legalEntities.find(e => e.value === value);
      toastProvider?.addToast({
        variant: 'info',
        title: 'Entity Selected',
        description: `Selected: ${entity?.label}`
      });
    }
  }

  function handleDeadlineChange(date: Date | undefined) {
    selectedDeadline = date;
    if (date) {
      const daysUntil = Math.ceil((date.getTime() - new Date().getTime()) / (1000 * 60 * 60 * 24));
      toastProvider?.showLegalDeadlineWarning('Discovery Deadline', daysUntil);
    }
  }

  const tabs = [
    { id: 'data-table', label: 'Data Table', icon: FileText },
    { id: 'combobox', label: 'Entity Selection', icon: Users },
    { id: 'date-picker', label: 'Date Picker', icon: Calendar },
    { id: 'timeline', label: 'Case Timeline', icon: Calendar },
    { id: 'custody', label: 'Chain of Custody', icon: Shield },
    { id: 'precedent', label: 'Legal Precedent', icon: Scale },
    { id: 'criminal', label: 'Criminal Profile', icon: AlertTriangle }
  ];
</script>

<svelte:head>
  <title>Legal Components Demo - Legal AI Platform</title>
  <meta name="description" content="Comprehensive demo of legal domain components with bits-ui primitives" />
</svelte:head>

<ToastProvider bind:this={toastProvider} />

<div class="min-h-screen bg-yorha-bg-primary">
  <!-- Navigation Breadcrumb -->
  <nav class="bg-yorha-bg-secondary border-b border-yorha-border px-6 py-3">
    <div class="max-w-7xl mx-auto">
      <div class="flex items-center space-x-2 text-sm font-mono text-yorha-text-secondary">
        <a href="/" class="hover:text-yorha-text-primary transition-colors">Home</a>
        <span>/</span>
        <a href="/demo" class="hover:text-yorha-text-primary transition-colors">Demo</a>
        <span>/</span>
        <span class="text-yorha-text-primary">Legal Components</span>
      </div>
    </div>
  </nav>

  <!-- Main Content -->
  <div class="max-w-7xl mx-auto p-6">
    <!-- Header -->
    <div class="text-center mb-8">
      <h1 class="text-3xl font-bold text-yorha-text-primary font-mono mb-2">
        Legal Components Demo
      </h1>
      <p class="text-yorha-text-secondary font-mono">
        Comprehensive showcase of legal domain components with bits-ui primitives
      </p>
    </div>

    <!-- Tab Navigation -->
    <div class="mb-8">
      <div class="flex flex-wrap gap-2 p-1 bg-yorha-bg-secondary rounded-lg border border-yorha-border">
        {#each tabs as tab}
          {@const IconComponent = tab.icon}
          <button
            onclick={() => activeTab = tab.id}
            class={`flex items-center gap-2 px-4 py-2 text-sm font-mono rounded transition-colors ${
              activeTab === tab.id
                ? 'bg-yorha-primary text-yorha-bg-primary'
                : 'text-yorha-text-secondary hover:text-yorha-text-primary hover:bg-yorha-bg-tertiary'
            }`}
          >
            <IconComponent class="w-4 h-4" />
            {tab.label}
          </button>
        {/each}
      </div>
    </div>

    <!-- Component Demos -->
    <div class="space-y-8">
      <!-- Data Table Demo -->
      {#if activeTab === 'data-table'}
        <section class="space-y-4">
          <h2 class="text-xl font-semibold text-yorha-text-primary font-mono">
            📊 Evidence Management Data Table
          </h2>
          <BitsDataTable
            data={evidenceData}
            columns={evidenceColumns}
            title="Evidence Inventory"
            searchable={true}
            filterable={true}
            exportable={true}
            selectable={true}
            onRowClick={handleRowClick}
            onExport={handleExport}
          />
        </section>
      {/if}

      <!-- Combobox Demo -->
      {#if activeTab === 'combobox'}
        <section class="space-y-4">
          <h2 class="text-xl font-semibold text-yorha-text-primary font-mono">
            🔍 Legal Entity Selection
          </h2>
          <div class="max-w-md">
            <BitsCombobox
              options={legalEntities}
              bind:value={selectedEntity}
              placeholder="Search legal entities..."
              searchPlaceholder="Type to search entities..."
              categories={true}
              creatable={true}
              label="Select Legal Entity"
              description="Choose from individuals, corporations, or government entities"
              onValueChange={handleEntityChange}
              onCreateOption={async (inputValue) => ({
                value: inputValue.toLowerCase().replace(/\s+/g, '_'),
                label: inputValue,
                category: 'Custom'
              })}
            />
            {#if selectedEntity}
              <div class="mt-4 p-3 bg-yorha-bg-secondary border border-yorha-border rounded">
                <p class="text-sm font-mono text-yorha-text-primary">
                  Selected Entity: <span class="font-medium">{selectedEntity}</span>
                </p>
              </div>
            {/if}
          </div>
        </section>
      {/if}

      <!-- Date Picker Demo -->
      {#if activeTab === 'date-picker'}
        <section class="space-y-4">
          <h2 class="text-xl font-semibold text-yorha-text-primary font-mono">
            📅 Legal Deadline Management
          </h2>
          <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <BitsDatePicker
              bind:value={selectedDeadline}
              label="Discovery Deadline"
              description="Set deadline for discovery phase completion"
              variant="deadline"
              onValueChange={handleDeadlineChange}
            />
            <BitsDatePicker
              label="Court Hearing Date"
              description="Schedule upcoming court appearance"
              variant="legal"
            />
            <BitsDatePicker
              label="Evidence Collection Date"
              description="Record when evidence was collected"
              variant="default"
              showTime={true}
            />
          </div>
          {#if selectedDeadline}
            <div class="mt-4 p-3 bg-yorha-bg-secondary border border-yorha-border rounded">
              <p class="text-sm font-mono text-yorha-text-primary">
                Discovery deadline set for: <span class="font-medium">{selectedDeadline.toLocaleDateString()}</span>
              </p>
            </div>
          {/if}
        </section>
      {/if}

      <!-- Case Timeline Demo -->
      {#if activeTab === 'timeline'}
        <section class="space-y-4">
          <h2 class="text-xl font-semibold text-yorha-text-primary font-mono">
            ⏱️ Case Timeline Visualization
          </h2>
          <CaseTimeline
            caseId="CASE001"
            caseName="State v. Digital Privacy Rights"
            events={caseTimelineEvents}
            showFutureEvents={true}
            interactive={true}
            onEventClick={(event) => toastProvider?.addToast({
              variant: 'info',
              title: 'Timeline Event',
              description: `Viewing: ${event.title}`
            })}
          />
        </section>
      {/if}

      <!-- Chain of Custody Demo -->
      {#if activeTab === 'custody'}
        <section class="space-y-4">
          <h2 class="text-xl font-semibold text-yorha-text-primary font-mono">
            🔒 Chain of Custody Tracking
          </h2>
          <ChainOfCustodyTracker
            evidence={evidenceItem}
            showFullHistory={true}
            interactive={true}
            onTransferEvidence={() => toastProvider?.addToast({
              variant: 'warning',
              title: 'Custody Transfer',
              description: 'Initiating evidence transfer process'
            })}
            onViewDetails={(transfer) => toastProvider?.addToast({
              variant: 'info',
              title: 'Transfer Details',
              description: `Viewing transfer: ${transfer.fromPerson} → ${transfer.toPerson}`
            })}
          />
        </section>
      {/if}

      <!-- Legal Precedent Demo -->
      {#if activeTab === 'precedent'}
        <section class="space-y-4">
          <h2 class="text-xl font-semibold text-yorha-text-primary font-mono">
            ⚖️ Legal Precedent Analysis
          </h2>
          <LegalPrecedentCard
            precedent={legalPrecedent}
            currentCaseId="CASE001"
            showRelevanceScore={true}
            showSimilarityScore={true}
            expandable={true}
            interactive={true}
            onViewFull={() => toastProvider?.addToast({
              variant: 'info',
              title: 'Precedent Details',
              description: 'Opening full precedent analysis'
            })}
            onAddToCase={() => toastProvider?.addToast({
              variant: 'success',
              title: 'Precedent Added',
              description: 'Legal precedent added to current case'
            })}
          />
        </section>
      {/if}

      <!-- Criminal Profile Demo -->
      {#if activeTab === 'criminal'}
        <section class="space-y-4">
          <h2 class="text-xl font-semibold text-yorha-text-primary font-mono">
            👤 Criminal Profile Management
          </h2>
          <CriminalProfile
            profile={criminalProfile}
            viewMode="full"
            showSensitiveInfo={false}
            interactive={true}
            onViewFullRecord={(recordId) => toastProvider?.addToast({
              variant: 'info',
              title: 'Criminal Record',
              description: `Viewing record: ${recordId}`
            })}
            onUpdateProfile={() => toastProvider?.addToast({
              variant: 'warning',
              title: 'Profile Update',
              description: 'Updating criminal profile information'
            })}
            onViewMugshot={() => toastProvider?.addToast({
              variant: 'info',
              title: 'Mugshot Viewer',
              description: 'Opening mugshot in viewer'
            })}
          />
        </section>
      {/if}
    </div>

    <!-- Component Status Footer -->
    <footer class="mt-12 pt-8 border-t border-yorha-border">
      <div class="text-center space-y-4">
        <h3 class="text-lg font-semibold text-yorha-text-primary font-mono">
          Component Library Status
        </h3>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-sm font-mono">
          <div class="bg-green-500/10 border border-green-500/20 rounded p-3">
            <div class="text-green-400 font-semibold">✅ All Components</div>
            <div class="text-yorha-text-secondary">Production ready</div>
          </div>
          <div class="bg-blue-500/10 border border-blue-500/20 rounded p-3">
            <div class="text-blue-400 font-semibold">🔧 Svelte 5</div>
            <div class="text-yorha-text-secondary">Full compatibility</div>
          </div>
          <div class="bg-purple-500/10 border border-purple-500/20 rounded p-3">
            <div class="text-purple-400 font-semibold">♿ Accessibility</div>
            <div class="text-yorha-text-secondary">WCAG 2.1 AA</div>
          </div>
          <div class="bg-yorha-primary/10 border border-yorha-primary/20 rounded p-3">
            <div class="text-yorha-primary font-semibold">⚖️ Legal Domain</div>
            <div class="text-yorha-text-secondary">Specialized components</div>
          </div>
        </div>
      </div>
    </footer>
  </div>
</div>

<style>
  :global(body) {
    font-family: 'Roboto Mono', monospace;
  }
</style>