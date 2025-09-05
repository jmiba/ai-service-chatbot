# Admin Interface - Library Contact Management

## Overview

The enhanced admin interface now provides comprehensive monitoring of both chat interactions and library staff contact requests through a unified dashboard.

## Access

- **URL**: `/view_logs` (requires admin authentication)
- **Authentication**: Use admin password configured in `secrets.toml`

## Features

### 📊 Two-Tab Interface

#### Tab 1: Chat Interaction Logs
- **Purpose**: Monitor regular chat interactions between users and AI
- **Data Source**: `log_table` database table
- **Features**:
  - Filter by error codes (E00, E01, E02)
  - View user queries and AI responses
  - See citation sources and metadata
  - Performance analytics (response times, confidence scores)

#### Tab 2: Library Contact Requests
- **Purpose**: Monitor escalated requests sent to library staff
- **Data Source**: `library_contacts` database table
- **Features**:
  - Summary metrics (total, sent, pending)
  - Ticket ID tracking (VIALIB-YYYYMMDD-XXXXXXXX format)
  - Email delivery status monitoring
  - SMTP error diagnostics

## Library Contact Request Details

### Status Types
- **✅ sent**: Email successfully delivered via SMTP
- **⚠️ template_only**: Email template created but not sent (SMTP not configured)
- **⏳ pending**: Request created but status unknown

### Information Displayed
- **Ticket ID**: Unique identifier for each request
- **Timestamp**: When the request was made
- **User Email**: Contact information (if provided)
- **Original Query**: User's research question
- **AI Response**: AI's attempt to answer (first 500 characters)
- **Technical Notes**: SMTP errors or delivery issues

## Administrative Actions

### Data Management
- **Delete Chat Logs**: Clear all interaction history
- **Delete Library Contacts**: Clear all contact request history
- **Selective Filtering**: View specific error types or time ranges

### Monitoring Capabilities
- **Real-time Updates**: Latest requests appear immediately
- **Status Tracking**: Monitor email delivery success/failure
- **Error Diagnostics**: SMTP configuration issues highlighted

## Database Schema

### library_contacts Table
```sql
CREATE TABLE library_contacts (
    id SERIAL PRIMARY KEY,
    ticket_id VARCHAR(50) UNIQUE NOT NULL,
    user_query TEXT NOT NULL,
    ai_response TEXT,
    user_email VARCHAR(255),
    timestamp TIMESTAMP NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    notes TEXT
);
```

## Use Cases

### For Library Administrators
1. **Monitor Request Volume**: Track how many users need human assistance
2. **Identify Common Issues**: See what topics frequently require escalation
3. **Email Delivery Health**: Ensure SMTP is working properly
4. **Response Time Tracking**: Monitor how quickly staff respond to tickets

### For Technical Administrators
1. **SMTP Diagnostics**: Identify email delivery problems
2. **System Performance**: Monitor AI response quality
3. **User Behavior**: Understand escalation patterns
4. **Database Health**: Track table growth and performance

## Best Practices

### Regular Monitoring
- Check library contacts daily during business hours
- Monitor SMTP status for delivery issues
- Review escalation patterns for system improvements

### Data Retention
- Archive old logs periodically to maintain performance
- Keep library contact history for accountability
- Export critical data before bulk deletions

### Security
- Limit admin access to authorized personnel only
- Monitor login attempts and admin activities
- Regularly update admin passwords

---

**Next Steps**: Consider adding email notifications for new library contact requests and automated SMTP health checks.
