"""
AI Chatbot Filter Examples and Implementation Guide
==================================================

This file demonstrates practical filter implementations for a university AI chatbot.
These filters enhance response quality, ensure appropriate content, and improve user experience.

"""

import re
import openai
from typing import Dict, List, Optional, Tuple

class ContentFilter:
    """Content filtering for AI chatbot responses"""
    
    def __init__(self, filter_settings: Dict):
        self.settings = filter_settings
        
    def apply_quality_filters(self, response: str, metadata: Dict) -> Tuple[str, bool, List[str]]:
        """
        Apply quality control filters to a response
        
        Returns:
            - filtered_response: The potentially modified response
            - should_block: Whether to block this response
            - warnings: List of warning messages
        """
        warnings = []
        should_block = False
        filtered_response = response
        
        # 1. Confidence threshold check
        confidence = metadata.get('confidence', 0.0)
        if confidence < self.settings.get('confidence_threshold', 0.7):
            warnings.append(f"Low confidence response ({confidence:.1%})")
            if confidence < 0.5:
                should_block = True
                
        # 2. Citation requirements
        citation_count = metadata.get('citation_count', 0)
        min_citations = self.settings.get('min_citations', 1)
        if citation_count < min_citations:
            warnings.append(f"Insufficient citations ({citation_count}/{min_citations})")
            
        # 3. Response length limits
        max_length = self.settings.get('max_response_length', 2000)
        if len(response) > max_length:
            filtered_response = response[:max_length] + "... [Response truncated]"
            warnings.append("Response truncated for length")
            
        return filtered_response, should_block, warnings
    
    def check_academic_integrity(self, user_input: str) -> Tuple[bool, str]:
        """
        Detect potential academic integrity violations
        
        Returns:
            - is_violation: Whether this might be a homework/exam question
            - guidance: Appropriate response guidance
        """
        homework_patterns = [
            r'solve.*problem.*for.*me',
            r'what.*is.*the.*answer.*to',
            r'help.*me.*with.*homework',
            r'assignment.*due.*tomorrow',
            r'exam.*question',
            r'quiz.*answer',
            r'write.*essay.*for.*me'
        ]
        
        for pattern in homework_patterns:
            if re.search(pattern, user_input.lower()):
                return True, "ACADEMIC_INTEGRITY_GUIDANCE"
                
        return False, "NORMAL"
    
    def detect_language(self, text: str) -> str:
        """Detect the language of input text"""
        # Simplified language detection
        german_words = ['der', 'die', 'das', 'und', 'ich', 'ist', 'mit', 'zu', 'ein', 'auf']
        english_words = ['the', 'and', 'is', 'to', 'in', 'it', 'you', 'that', 'he', 'was']
        
        text_lower = text.lower()
        german_count = sum(1 for word in german_words if word in text_lower)
        english_count = sum(1 for word in english_words if word in text_lower)
        
        if german_count > english_count:
            return 'de'
        else:
            return 'en'
    
    def check_topic_appropriateness(self, user_input: str, topic_restriction: str) -> Tuple[bool, str]:
        """
        Check if the topic is appropriate for a university chatbot
        
        Returns:
            - is_appropriate: Whether the topic is allowed
            - reason: Explanation of the decision
        """
        if topic_restriction == "None":
            return True, "No restrictions"
            
        university_keywords = [
            'study', 'course', 'professor', 'lecture', 'exam', 'semester',
            'studium', 'kurs', 'vorlesung', 'prüfung', 'universität',
            'research', 'thesis', 'academic', 'library', 'campus'
        ]
        
        inappropriate_keywords = [
            'politics', 'religion', 'personal advice', 'medical diagnosis',
            'politik', 'religion', 'persönlicher rat', 'medizinische diagnose'
        ]
        
        input_lower = user_input.lower()
        
        # Check for inappropriate content
        for keyword in inappropriate_keywords:
            if keyword in input_lower:
                return False, f"Topic not appropriate for university context: {keyword}"
        
        # Check for university-related content
        if topic_restriction in ["University-related only", "Academic only"]:
            has_university_context = any(keyword in input_lower for keyword in university_keywords)
            if not has_university_context:
                return False, "Topic should be university or academic related"
                
        return True, "Topic is appropriate"

class ResponseFormatter:
    """Format responses according to university standards"""
    
    def __init__(self, citation_style: str = "Academic (APA-style)"):
        self.citation_style = citation_style
    
    def format_citations(self, response: str, citations: List[Dict]) -> str:
        """Format citations according to the specified style"""
        
        if self.citation_style == "Academic (APA-style)":
            return self._format_apa_style(response, citations)
        elif self.citation_style == "Numbered":
            return self._format_numbered(response, citations)
        elif self.citation_style == "Simple links":
            return self._format_simple_links(response, citations)
        else:  # Inline
            return self._format_inline(response, citations)
    
    def _format_apa_style(self, response: str, citations: List[Dict]) -> str:
        """Format citations in APA style"""
        formatted_response = response
        
        if citations:
            formatted_response += "\n\n**Quellen:**\n"
            for i, citation in enumerate(citations, 1):
                title = citation.get('title', 'Untitled')
                url = citation.get('url', '#')
                formatted_response += f"{i}. {title}. Verfügbar unter: {url}\n"
                
        return formatted_response
    
    def _format_numbered(self, response: str, citations: List[Dict]) -> str:
        """Format with numbered references"""
        # Implementation for numbered citations
        return response  # Simplified
    
    def _format_simple_links(self, response: str, citations: List[Dict]) -> str:
        """Format with simple links"""
        # Implementation for simple links
        return response  # Simplified
    
    def _format_inline(self, response: str, citations: List[Dict]) -> str:
        """Format with inline citations"""
        # Implementation for inline citations
        return response  # Simplified

class UserTypeAdapter:
    """Adapt responses based on user type and context"""
    
    def adapt_response(self, response: str, user_type: str) -> str:
        """Adapt response style based on user type"""
        
        if user_type == "Student-friendly":
            return self._make_student_friendly(response)
        elif user_type == "Faculty-focused":
            return self._make_faculty_focused(response)
        elif user_type == "Staff-oriented":
            return self._make_staff_oriented(response)
        else:  # Standard
            return response
    
    def _make_student_friendly(self, response: str) -> str:
        """Make response more accessible for students"""
        # Add explanatory notes, simpler language
        if len(response) > 500:
            response = "**Kurze Antwort:** " + response[:200] + "...\n\n**Ausführliche Erklärung:**\n" + response
        return response
    
    def _make_faculty_focused(self, response: str) -> str:
        """Focus on academic and research aspects"""
        # Add research context, academic references
        return f"**Akademischer Kontext:**\n{response}"
    
    def _make_staff_oriented(self, response: str) -> str:
        """Focus on administrative and practical aspects"""
        # Emphasize procedures, regulations, practical steps
        return f"**Verwaltungshinweis:**\n{response}"

# Example filter implementation in main application
def apply_comprehensive_filters(user_input: str, ai_response: str, metadata: Dict, filter_settings: Dict) -> Dict:
    """
    Apply all filters to a user input and AI response
    
    Returns a comprehensive filtering result
    """
    
    # Initialize filters
    content_filter = ContentFilter(filter_settings)
    formatter = ResponseFormatter(filter_settings.get('citation_style', 'Academic (APA-style)'))
    adapter = UserTypeAdapter()
    
    result = {
        'original_response': ai_response,
        'filtered_response': ai_response,
        'should_block': False,
        'warnings': [],
        'modifications': [],
        'user_guidance': None
    }
    
    # 1. Check academic integrity
    is_violation, guidance = content_filter.check_academic_integrity(user_input)
    if is_violation:
        result['user_guidance'] = guidance
        result['warnings'].append("Potential academic integrity concern")
    
    # 2. Check topic appropriateness
    is_appropriate, reason = content_filter.check_topic_appropriateness(
        user_input, 
        filter_settings.get('topic_restriction', 'University-related only')
    )
    if not is_appropriate:
        result['should_block'] = True
        result['warnings'].append(f"Topic restriction: {reason}")
    
    # 3. Apply quality filters
    filtered_response, should_block_quality, quality_warnings = content_filter.apply_quality_filters(
        ai_response, metadata
    )
    result['filtered_response'] = filtered_response
    result['should_block'] = result['should_block'] or should_block_quality
    result['warnings'].extend(quality_warnings)
    
    # 4. Format citations
    citations = metadata.get('citations', [])
    if citations:
        result['filtered_response'] = formatter.format_citations(result['filtered_response'], citations)
        result['modifications'].append("Citations formatted")
    
    # 5. Adapt for user type
    user_type = filter_settings.get('user_type_adaptation', 'Standard')
    if user_type != 'Standard':
        result['filtered_response'] = adapter.adapt_response(result['filtered_response'], user_type)
        result['modifications'].append(f"Adapted for {user_type}")
    
    # 6. Language consistency check
    if filter_settings.get('language_consistency', True):
        input_lang = content_filter.detect_language(user_input)
        response_lang = content_filter.detect_language(ai_response)
        if input_lang != response_lang:
            result['warnings'].append(f"Language mismatch: Input ({input_lang}) vs Response ({response_lang})")
    
    return result

# Usage example
if __name__ == "__main__":
    # Example filter settings (these would come from your database)
    sample_filter_settings = {
        'confidence_threshold': 0.7,
        'min_citations': 1,
        'max_response_length': 2000,
        'enable_fact_checking': True,
        'academic_integrity_check': True,
        'language_consistency': True,
        'topic_restriction': 'University-related only',
        'inappropriate_content_filter': True,
        'user_type_adaptation': 'Student-friendly',
        'citation_style': 'Academic (APA-style)',
    }
    
    # Example usage
    user_question = "Kannst du mir bei meiner Hausaufgabe helfen?"
    ai_response = "Gerne kann ich Ihnen beim Verständnis der Konzepte helfen..."
    metadata = {
        'confidence': 0.85,
        'citation_count': 2,
        'citations': [
            {'title': 'Universitäts-Leitfaden', 'url': 'https://europa-uni.de/guide'},
            {'title': 'Studienordnung', 'url': 'https://europa-uni.de/regulations'}
        ]
    }
    
    result = apply_comprehensive_filters(user_question, ai_response, metadata, sample_filter_settings)
    
    print("Filter Result:")
    print(f"Should block: {result['should_block']}")
    print(f"Warnings: {result['warnings']}")
    print(f"Modifications: {result['modifications']}")
    print(f"Filtered response: {result['filtered_response'][:200]}...")
