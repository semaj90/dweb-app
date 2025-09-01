
/**
 * Demo Data Generator for Legal AI System Testing
 */

export interface DemoCase {
  id: string;
  title: string;
  description: string;
  status: "active" | "pending" | "closed";
  priority: "low" | "medium" | "high" | "urgent";
  createdAt: Date;
  updatedAt: Date;
  assignedTo?: string;
  tags: string[];
}

export interface DemoEvidence {
  id: string;
  caseId: string;
  title: string;
  description: string;
  type:
    | "police_report"
    | "witness_statement"
    | "financial_records"
    | "digital_forensics"
    | "physical_evidence"
    | "expert_testimony";
  status: "new" | "reviewing" | "approved";
  content: string;
  uploadedAt: Date;
  fileSize: number;
  tags: string[];
}

export interface DemoPerson {
  id: string;
  name: string;
  role: "suspect" | "witness" | "victim" | "officer" | "expert" | "other";
  contactInfo: {
    phone?: string;
    email?: string;
    address?: string;
  };
  notes: string;
}

class DemoDataGenerator {
  private caseCounter = 1;
  private evidenceCounter = 1;
  private personCounter = 1;

  /**
   * Generate sample cases
   */
  generateCases(count: number = 5): DemoCase[] {
    const caseTemplates = [
      {
        title: "State v. Johnson - Embezzlement Investigation",
        description:
          "Corporate embezzlement investigation involving $2.3M in misappropriated funds. Multiple financial institutions affected.",
        priority: "high" as const,
        tags: ["embezzlement", "corporate", "financial-crimes"],
      },
      {
        title: "People v. Martinez - Assault Case",
        description:
          "Aggravated assault incident outside downtown restaurant. Multiple witnesses, security footage available.",
        priority: "medium" as const,
        tags: ["assault", "public-incident", "witnesses"],
      },
      {
        title: "State v. Chen - Identity Theft Ring",
        description:
          "Multi-state identity theft operation targeting elderly victims. Complex digital evidence trail.",
        priority: "urgent" as const,
        tags: ["identity-theft", "cybercrime", "elderly-victims"],
      },
      {
        title: "People v. Anderson - DUI Vehicular Homicide",
        description:
          "Fatal traffic collision involving suspected DUI. Blood analysis and accident reconstruction required.",
        priority: "high" as const,
        tags: ["dui", "vehicular-homicide", "traffic-fatality"],
      },
      {
        title: "State v. Williams - Drug Distribution",
        description:
          "Large-scale narcotics distribution network. Surveillance operations and controlled purchases documented.",
        priority: "medium" as const,
        tags: ["drug-crimes", "distribution", "surveillance"],
      },
    ];

    return Array.from({ length: count }, (_, i) => {
      const template = caseTemplates[i % caseTemplates.length];
      const createdAt = new Date(
        Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000,
      ); // Last 30 days

      return {
        id: `CASE-2024-${(this.caseCounter++).toString().padStart(3, "0")}`,
        ...template,
        status: ["active", "pending", "closed"][
          Math.floor(Math.random() * 3)
        ] as any,
        createdAt,
        updatedAt: new Date(
          createdAt.getTime() + Math.random() * 7 * 24 * 60 * 60 * 1000,
        ),
        assignedTo: ["Detective Smith", "Prosecutor Jones", "Officer Brown"][
          Math.floor(Math.random() * 3)
        ],
      };
    });
  }

  /**
   * Generate sample evidence for a case
   */
  generateEvidence(caseId: string, count: number = 8): DemoEvidence[] {
    const evidenceTemplates = [
      {
        title: "Initial Police Report",
        type: "police_report" as const,
        content: `INCIDENT REPORT - Case ${caseId}

Date: ${new Date().toLocaleDateString()}
Reporting Officer: Officer Martinez, Badge #1247
Location: 1425 Main Street, Downtown District

INCIDENT SUMMARY:
On the above date and time, I responded to a call regarding suspicious activity at the above location. Upon arrival, I observed several individuals engaged in what appeared to be an illegal transaction.

INDIVIDUALS INVOLVED:
- Primary Suspect: John Doe, Male, Age 34, 5'10", Brown hair, Blue eyes
- Witness: Jane Smith, Female, Age 28, Resident of nearby apartment
- Additional witnesses present but not yet interviewed

EVIDENCE COLLECTED:
- Digital photographs of the scene
- Physical evidence secured in evidence locker #247
- Witness statements recorded (see attached)

NARRATIVE:
At approximately 14:30 hours, I was dispatched to investigate reports of suspicious activity. The complainant, later identified as Jane Smith, reported observing what she believed to be a drug transaction in the alley behind 1425 Main Street.

Upon my arrival, I positioned my patrol vehicle to observe the area. I witnessed two individuals, later identified as John Doe and an unknown male subject, engaged in what appeared to be an exchange of money and a small package.

I approached the subjects and identified myself as a police officer. The unknown subject immediately fled on foot southbound on Main Street. John Doe remained at the scene and was cooperative with the investigation.

A search incident to arrest revealed:
- $847 in small denomination bills
- Small plastic baggie containing white crystalline substance (field tested positive for cocaine)
- Digital scale
- Multiple cell phones

RECOMMENDATIONS:
1. Laboratory analysis of suspected narcotics
2. Follow-up interview with witness Jane Smith
3. Canvass area for additional witnesses
4. Analysis of cell phone data pending warrant

This case requires further investigation to identify the fleeing suspect and determine the scope of any ongoing criminal activity.

Report submitted by: Officer Martinez
Supervisor Review: Sergeant Johnson
Date: ${new Date().toLocaleDateString()}`,
        tags: ["initial-report", "arrest", "evidence-collection"],
      },
      {
        title: "Witness Statement - Jane Smith",
        type: "witness_statement" as const,
        content: `WITNESS STATEMENT

Case Number: ${caseId}
Date: ${new Date().toLocaleDateString()}
Time: 15:45 hours
Location: Police Station, Interview Room 3

WITNESS INFORMATION:
Name: Jane Elizabeth Smith
Address: 1427 Main Street, Apt 3B
Phone: (555) 123-4567
Date of Birth: 03/15/1995
Occupation: Graphic Designer

STATEMENT:
My name is Jane Smith, and I live at 1427 Main Street, apartment 3B. I am providing this statement voluntarily regarding what I witnessed on [DATE] at approximately 2:30 PM.

I was in my apartment working from home when I heard loud voices coming from the alley behind my building. This is unusual because it's normally a quiet area. I looked out my window, which faces the alley, and saw two men standing near the dumpster.

One man was tall, maybe 6 feet, wearing a red hoodie and jeans. The other was shorter, about 5'8", wearing a black jacket. I couldn't see their faces clearly from my window, but I could tell they were having some kind of argument.

The man in the red hoodie kept looking around, like he was nervous about something. He pulled out what looked like a small bag from his pocket and handed it to the other man. In return, the man in the black jacket gave him what appeared to be money.

I thought this looked suspicious, so I decided to call the police. While I was on the phone with 911, a police car arrived. When the officer got out of his car, the man in the red hoodie ran away really fast toward Main Street. The other man just stood there.

I stayed at my window and watched the officer talk to the man who didn't run. The officer searched him and found some things, but I couldn't see exactly what from my apartment.

After about 20 minutes, the officer came to my door and asked if I had seen anything. I told him what I just told you, and he asked me to come to the station to give this formal statement.

I am certain about what I saw. The exchange definitely looked like some kind of drug deal to me. I've lived in this neighborhood for three years, and I've never seen anything like this before.

I am willing to testify in court if needed.

STATEMENT CONCLUSION:
This statement was given voluntarily. I have read this statement, and it is true and accurate to the best of my knowledge.

Witness Signature: _________________ Date: _______
Officer: Detective Brown, Badge #3451
Case Status: Open Investigation`,
        tags: ["witness", "drug-transaction", "firsthand-account"],
      },
      {
        title: "Financial Records Analysis",
        type: "financial_records" as const,
        content: `FINANCIAL ANALYSIS REPORT
Case: ${caseId}
Analyst: Forensic Accountant Sarah Chen, CPA
Date: ${new Date().toLocaleDateString()}

EXECUTIVE SUMMARY:
Analysis of banking records for suspect John Doe reveals suspicious financial activity consistent with narcotics trafficking over a 6-month period.

ACCOUNTS ANALYZED:
- First National Bank, Account #****7892
- Community Credit Union, Account #****3456
- Cash App, Account: john.doe.1989

KEY FINDINGS:

1. CASH DEPOSIT PATTERNS:
   - 47 cash deposits ranging from $200-$800
   - Deposits made at multiple branch locations
   - Timing correlates with surveillance dates
   - Total cash deposits: $23,750 over 6 months

2. UNUSUAL TRANSACTION ACTIVITY:
   - Frequent small withdrawals immediately after deposits
   - Multiple peer-to-peer payment app transactions
   - Payments to individuals with criminal histories
   - Geographic pattern matching known drug activity areas

3. LIFESTYLE INCONSISTENCY:
   - Reported income: $28,000/year (part-time retail)
   - Bank activity suggests additional income source
   - Large cash purchases not consistent with legitimate income

SUPPORTING EVIDENCE:
- Bank statements (Jan-June 2024)
- ATM surveillance footage
- Deposit slip analysis
- Cross-reference with DEA database

MONEY LAUNDERING INDICATORS:
✓ Structuring deposits below $10,000 reporting threshold
✓ Use of multiple accounts and institutions
✓ Geographic dispersion of transactions
✓ Inconsistent income sources

RECOMMENDATIONS:
1. Asset forfeiture analysis
2. Additional account discovery
3. Wire transfer investigation
4. Coordination with DEA Financial Crimes Unit

This analysis supports charges of money laundering in addition to narcotics trafficking.

Prepared by: Sarah Chen, CPA, CFE
Reviewed by: Supervisor Johnson
Distribution: Prosecutor Williams, Detective Brown`,
        tags: ["financial-crimes", "money-laundering", "forensic-accounting"],
      },
      {
        title: "Digital Forensics Report",
        type: "digital_forensics" as const,
        content: `DIGITAL FORENSICS EXAMINATION REPORT

Case Number: ${caseId}
Examiner: Tech Specialist Maria Rodriguez
Badge/ID: DF-2847
Date of Report: ${new Date().toLocaleDateString()}

EVIDENCE INFORMATION:
Item #: 2024-${caseId}-001
Description: Apple iPhone 12 Pro, Black, 128GB
Serial Number: G6QZ3L9KN72F
IMEI: 356789101234567
Condition: Good, screen locked with passcode

EXAMINATION SUMMARY:
Complete forensic examination of suspect's mobile device using Cellebrite UFED 4PC and Oxygen Detective Suite. Legal authority provided via search warrant #SW-2024-1847.

TECHNICAL DETAILS:
- Device successfully bypassed using GrayKey solution
- Full file system extraction completed
- 47,892 total items recovered
- Hash verification: MD5 a7b3c9d8e2f1g4h5i6j7k8l9m0n1o2p3

KEY FINDINGS:

1. COMMUNICATIONS EVIDENCE:
   Text Messages:
   - 2,847 text messages analyzed
   - 23 messages contain drug-related terminology
   - Deleted messages recovered from unallocated space
   
   Call Logs:
   - 156 outgoing calls to known associates
   - Pattern of short calls (avg 45 seconds)
   - Burner phone numbers identified

2. INCRIMINATING CONTENT:
   Photos (347 total):
   - Images of currency bundles
   - Photos of suspected narcotics
   - Pictures of drug paraphernalia
   - Location metadata preserved
   
   Notes Application:
   - Coded customer list with quantities
   - Financial tracking records
   - Territory/location assignments

3. LOCATION DATA:
   - GPS coordinates correlate with surveillance locations
   - Frequent visits to known drug activity areas
   - Timeline matches witness statements
   - Google Maps searches for secure meeting locations

4. FINANCIAL APPLICATIONS:
   - Cash App: $12,450 in transactions
   - Venmo: Multiple payments with drug emojis
   - Calculator app used for pricing calculations
   - Cryptocurrency wallet app (Bitcoin)

5. SOCIAL MEDIA EVIDENCE:
   - Instagram posts showing lifestyle inconsistent with reported income
   - Snapchat messages with drug references (recovered from cache)
   - Facebook Messenger contacts include known dealers

DELETED DATA RECOVERY:
- 127 deleted text messages recovered
- 89 deleted photos showing evidence of drug activity
- Browser history showing research on avoiding police surveillance
- Deleted contacts linked to criminal database

CONCLUSIONS:
The examination of this device provides substantial evidence supporting charges of narcotics distribution, money laundering, and criminal conspiracy. The combination of communications, financial records, and location data creates a comprehensive digital footprint of criminal activity.

CHAIN OF CUSTODY:
Evidence maintained in secure forensics lab throughout examination. All procedures followed DOJ Digital Forensics guidelines.

ATTACHMENTS:
- Full forensic report (247 pages)
- Extracted data files
- Hash verification records
- Legal authority documentation

Report Prepared: Maria Rodriguez, Digital Forensics Specialist
Quality Review: Supervisor Chen
Legal Review: Prosecutor Williams`,
        tags: ["digital-forensics", "mobile-device", "recovered-data"],
      },
    ];

    return Array.from({ length: count }, (_, i) => {
      const template = evidenceTemplates[i % evidenceTemplates.length];
      const uploadedAt = new Date(
        Date.now() - Math.random() * 14 * 24 * 60 * 60 * 1000,
      ); // Last 14 days

      return {
        id: `EVD-${caseId}-${(this.evidenceCounter++).toString().padStart(3, "0")}`,
        caseId,
        ...template,
        description: `${template.type.replace("_", " ")} evidence for case ${caseId}`,
        status: ["new", "reviewing", "approved"][
          Math.floor(Math.random() * 3)
        ] as any,
        uploadedAt,
        fileSize: Math.floor(Math.random() * 5000000) + 10000, // 10KB to 5MB
        tags: [...template.tags, "demo-data"],
      };
    });
  }

  /**
   * Generate sample persons of interest
   */
  generatePersons(caseId: string, count: number = 6): DemoPerson[] {
    const personTemplates = [
      {
        name: "John Michael Doe",
        role: "suspect" as const,
        contactInfo: {
          phone: "(555) 234-5678",
          address: "742 Elm Street, Apt 2B",
        },
        notes:
          "Primary suspect in narcotics investigation. Known associates include several individuals with prior drug convictions.",
      },
      {
        name: "Jane Elizabeth Smith",
        role: "witness" as const,
        contactInfo: {
          phone: "(555) 123-4567",
          email: "jane.smith@email.com",
          address: "1427 Main Street, Apt 3B",
        },
        notes:
          "Key eyewitness to alleged drug transaction. Cooperative and credible witness.",
      },
      {
        name: "Officer Robert Martinez",
        role: "officer" as const,
        contactInfo: {
          phone: "(555) 987-6543",
        },
        notes:
          "Arresting officer. Badge #1247. 8 years experience in narcotics enforcement.",
      },
      {
        name: "Detective Sarah Brown",
        role: "officer" as const,
        contactInfo: {
          phone: "(555) 555-0123",
        },
        notes:
          "Lead detective on case. Conducted follow-up interviews and evidence analysis.",
      },
      {
        name: "Dr. Michael Chen",
        role: "expert" as const,
        contactInfo: {
          phone: "(555) 444-5555",
          email: "mchen@forensicslab.gov",
        },
        notes:
          "Forensic chemist. Analyzed suspected narcotics and provided expert testimony.",
      },
      {
        name: "Maria Rodriguez",
        role: "expert" as const,
        contactInfo: {
          phone: "(555) 333-7777",
          email: "mrodriguez@digitalforensics.gov",
        },
        notes:
          "Digital forensics specialist. Examined mobile devices and recovered deleted data.",
      },
    ];

    return Array.from({ length: count }, (_, i) => {
      const template = personTemplates[i % personTemplates.length];

      return {
        id: `PER-${caseId}-${(this.personCounter++).toString().padStart(3, "0")}`,
        ...template,
      };
    });
  }

  /**
   * Generate a complete case with all related data
   */
  generateCompleteCase(): {
    case: DemoCase;
    evidence: DemoEvidence[];
    persons: DemoPerson[];
  } {
    const cases = this.generateCases(1);
    const caseData = cases[0];

    return {
      case: caseData,
      evidence: this.generateEvidence(caseData.id, 4),
      persons: this.generatePersons(caseData.id, 4),
    };
  }

  /**
   * Generate sample analysis results for testing
   */
  generateAnalysisResults(caseId: string, evidenceId: string) {
    return {
      sessionId: `analysis_${caseId}_${Date.now()}`,
      status: "completed",
      step: "case_synthesis",
      outputs: {
        evidence_analysis: {
          documentType: "police_report",
          keyFacts: [
            "Drug transaction observed by witness Jane Smith",
            "Suspect John Doe arrested with narcotics and cash",
            "Co-conspirator fled scene on foot",
            "Physical evidence secured in police custody",
          ],
          timelineEvents: [
            {
              date: "2024-01-15",
              time: "14:30",
              event: "Initial drug transaction observed",
            },
            {
              date: "2024-01-15",
              time: "14:35",
              event: "Police arrived on scene",
            },
            {
              date: "2024-01-15",
              time: "14:40",
              event: "Suspect arrested, co-conspirator fled",
            },
            {
              date: "2024-01-15",
              time: "15:00",
              event: "Evidence secured and processed",
            },
          ],
          evidenceItems: [
            "$847 in cash",
            "Suspected cocaine (field tested positive)",
            "Digital scale",
            "Multiple cell phones",
          ],
          concerns: [
            "Co-conspirator still at large",
            "Need laboratory confirmation of narcotics",
            "Require additional witness interviews",
          ],
          confidence: 0.92,
        },
        persons_extracted: {
          persons: [
            {
              personId: "per_001",
              fullName: "John Michael Doe",
              aliases: ["Johnny D"],
              role: "suspect",
              contactInfo: {
                address: "742 Elm Street, Apt 2B",
              },
              physicalDescription:
                "Male, Age 34, 5'10\", Brown hair, Blue eyes",
              organizations: [],
              firstMentioned: "police_report_initial",
            },
            {
              personId: "per_002",
              fullName: "Jane Elizabeth Smith",
              aliases: [],
              role: "witness",
              contactInfo: {
                phone: "(555) 123-4567",
                address: "1427 Main Street, Apt 3B",
              },
              physicalDescription: "Female, Age 28",
              organizations: [],
              firstMentioned: "witness_statement",
            },
          ],
          relationships: [
            {
              person1Id: "per_001",
              person2Id: "per_002",
              relationshipType: "witness_to",
              description:
                "Jane Smith witnessed John Doe's alleged criminal activity",
              strength: "strong",
              evidence: "eyewitness testimony and police report",
            },
          ],
          summary: {
            totalPersons: 2,
            totalRelationships: 1,
            keyFigures: ["per_001", "per_002"],
            missingInfo: ["identity of fleeing co-conspirator"],
          },
        },
        neo4j_updates: {
          cypherQueries: [
            "MERGE (p:Person {name: 'John Michael Doe'}) SET p.age = 34, p.role = 'suspect'",
            "MERGE (p:Person {name: 'Jane Elizabeth Smith'}) SET p.age = 28, p.role = 'witness'",
            "MATCH (s:Person {name: 'John Michael Doe'}), (w:Person {name: 'Jane Elizabeth Smith'}) MERGE (w)-[:WITNESSED]->(s)",
          ],
          nodesCreated: 2,
          relationshipsCreated: 1,
          conflicts: [],
          success: true,
        },
        case_synthesis: {
          caseStrength: "strong",
          keyFindings: [
            "Direct eyewitness testimony from credible witness",
            "Physical evidence recovered from suspect",
            "Suspect cooperative during arrest",
          ],
          evidenceAnalysis: {
            strengths: [
              "Eyewitness testimony from Jane Smith",
              "Physical narcotics evidence",
              "Large amount of cash consistent with drug sales",
              "Digital evidence on seized phones",
            ],
            weaknesses: [
              "Co-conspirator still unidentified",
              "No direct observation of money exchange by officer",
              "Field test needs laboratory confirmation",
            ],
            gaps: [
              "Laboratory analysis of suspected narcotics",
              "Digital forensics of seized phones",
              "Canvas for additional witnesses",
            ],
          },
          personNetworkAnalysis: {
            centralFigures: ["John Michael Doe"],
            suspiciousConnections: [
              "Multiple phone numbers suggest larger operation",
            ],
            investigationPriorities: [
              "Identify and apprehend fleeing co-conspirator",
              "Analyze phone records for additional suspects",
            ],
          },
          legalStrategy: {
            viableCharges: [
              "Possession with Intent to Distribute",
              "Drug Distribution",
              "Money Laundering",
            ],
            evidenceRequirements: [
              "Laboratory confirmation of narcotics",
              "Digital forensics report",
              "Financial records analysis",
            ],
            risks: [
              "Defense may challenge witness credibility",
              "Search and seizure issues if warrant improper",
            ],
            timeline:
              "Ready for preliminary hearing in 3-4 weeks pending lab results",
          },
          nextSteps: [
            "Submit evidence to laboratory for analysis",
            "Execute search warrant for suspect's residence",
            "Interview additional potential witnesses",
            "Coordinate with DEA for larger investigation",
          ],
          confidence: 0.87,
        },
      },
    };
  }
}

// Export singleton instance
export const demoDataGenerator = new DemoDataGenerator();
;
// Export utility functions
export function createTestEvidence(): string {
  return `POLICE INCIDENT REPORT - CASE DEMO-001

Date: ${new Date().toLocaleDateString()}
Time: 14:30 Hours
Location: 1425 Main Street, Downtown District
Reporting Officer: Martinez, Roberto - Badge #1247

INCIDENT SUMMARY:
Responded to call regarding suspicious activity. Observed apparent drug transaction between two male subjects. One suspect apprehended, second suspect fled scene.

EVIDENCE RECOVERED:
- $847 USD in small denomination bills
- Plastic baggie containing white powder (field tested positive for cocaine)
- Digital scale
- Two cellular phones

WITNESS:
Jane Smith (DOB: 03/15/1995)
Address: 1427 Main Street, Apt 3B
Phone: (555) 123-4567

Witness observed transaction from apartment window and called 911.

SUSPECT INFORMATION:
John Doe (DOB: 08/22/1989)
Address: 742 Elm Street, Apt 2B
Charges: Possession with Intent to Distribute

RECOMMENDATIONS:
1. Laboratory analysis of suspected narcotics
2. Digital forensics on seized phones
3. Follow-up witness interview
4. BOLO for fleeing suspect (Male, Red hoodie, approximately 6'0")

Report Status: Preliminary
Supervisor Review: Pending

This is demonstration evidence for testing the multi-agent analysis system.`;
}
