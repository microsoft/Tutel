gemv_nt_bf16xfp8_block_v2|threadIdx.x=64;threadIdx.y=1;blockIdx.x=2112;$=4;$$=1;$0=1;$1=-1;$2=-1;$3=1|arg_0=1:0:int32;arg_1=1:1:int32;arg_2=0:1:int32;arg_3=0:0:int32;arg_4=2:0:int32;arg_5=2:1:int32;o_type=infer:bfloat16:#3,#0|2:bfloat2x16,2:float4x32,2:float32|c-rocm||ELF@        �            @       (      L  @ 8 	 @         @       @       @       �      �                                                                                           �      �                   �      �6      �6      p       @	                   0      0G      0G                                  �      �6      �6      p       p              R�td   �      �6      �6      p       @	             Q�td                                                         8      8      8      D      D                0      AMDGPU  ��amdhsa.kernels�� �.agpr_count �.args���.actual_access�read_only�.address_space�global�.offset �.size�.value_kind�global_buffer��.actual_access�read_only�.address_space�global�.offset�.size�.value_kind�global_buffer��.actual_access�read_only�.address_space�global�.offset�.size�.value_kind�global_buffer��.actual_access�write_only�.address_space�global�.offset�.size�.value_kind�global_buffer��.offset �.size�.value_kind�by_value��.offset$�.size�.value_kind�by_value��.offset(�.size�.value_kind�by_value��.offset,�.size�.value_kind�by_value��.offset0�.size�.value_kind�by_value��.offset4�.size�.value_kind�by_value�.group_segment_fixed_size� �.kernarg_segment_align�.kernarg_segment_size8�.language�OpenCL C�.language_version� �.max_flat_workgroup_size@�.name�gemv_nt_bf16xfp8_block_v2�.private_segment_fixed_size �.sgpr_count�.sgpr_spill_count �.symbol�gemv_nt_bf16xfp8_block_v2.kd�.uniform_work_group_size�.uses_dynamic_stack«.vgpr_count,�.vgpr_spill_count �.wavefront_size@�amdhsa.target�)amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-�amdhsa.version�                                       �
          �      @       8    
 0G                            ��   �S��vϾs��-                                    gemv_nt_bf16xfp8_block_v2 gemv_nt_bf16xfp8_block_v2.kd __hip_cuid_2751b0aab6457f19                                                                    8       @                          
   � � �          
�    ��4   �����������~���� 
�     �   G~ �� )� ���
��O~  ��$~�����������	�����	�������� h� �� ��  ~������� &�  ����(�   �� h�(�   � h�( �� � $hh ��h�"~ ��$h�"�"�
"��� �	)  �1  �!  �P�  �\�  �\� �"�"�" � ��h�q���(>&  ��p���0
&��  �@~�(<$�0
  �� =�*B&  ���*@$�(~�2
&��  @��Az�@~�,>&  ���,<$�2
 @�� =R�.>&  ���.<$�,~�4
&��  @��=R�\� �<~�4
 p���(:&  ���(8$@��9b�*:&  ���*8$�(~�6
&��  @��9b�8~�,2&  ���,0$�6
 @��1R�.2&  ���.0$�,~@��1R  ��)
�(h   �(*" �! �(h�(*" �	)  �
1 "v �P�  �\�  �\� �hq���(>&  ��p���0
&��  �@~�(<$�0
  �� =�*B&  ���*@$�(~�2
&��  @��Az�@~�,>&  ���,<$�2
 @�� =R�.>&  ���.<$�,~�4
&��  @��=R�\� �<~�4
 p���(:&  ���(8$@��9b�*:&  ���*8$�(~�6
&��  @��9b�8~�,2&  ���,0$�6
 @��1R�.2&  ���.0$�,~@��1R  ��)
�(h   �(*" �! �(h�   �(*" �	)  �1 "v �P�  �\�  �\� �h�   �" �	) �hq���(>&  ��p���0
&��  �@~�(<$�0
  �� =�*B&  ���*@$�(~�2
&��  @��Az�@~�,>&  ���,<$�2
 @�� =R�.>&  ���.<$�,~�4
&��  @��=R�\� �<~�4
 p���(:&  ���(8$@��9b�*:&  ���*8$�(~�6
&��  @��9b�8~�,2&  ���,0$�6
 @��1R�.2&  ���.0$�,~@��1R  ��)
�(h   �(*" �!  �1 "v �P�  �\�  �\� �h   �hq���(>&  ��p���0
&��  �~�(<$�0
  ��=�*>&  ���*<$�(~�2
&��  @��=2�<~�,*&  ���,($�2
 @��)2�.*&  ���.($�,~�4
&��  @��)2�\� �8~�4
 p���(2&  ���(0$@��12�*2&  ���*0$�(~�6
&��  @��12�0~�,*&  ���,($�6
 @��)2�.*&  ���.($�,~@��)2  ��
"v���� �� ��� ��� &�  �"~  �   �o ���������h�  ��� hh ��~�����  h   ��h\ ��h�}j ������$h� "�$&"���
 � !  �	)  �\�
 �\�
 
 �\� �" �1  �P� s���.&  ���,$q���$&��  �2&  ���0$�&  ���$�6&  ���4$�&  ���$�:&  ���8$�&  ���$�$
 �&&��  �&< �$~�(H&��  �(L �*P&��  �*T �(~�@~�D~$�H~ ��-&�L~@��1J(�P~@�� J*�T~@��"5�>&  ��@��$�<$@��&9  ��@��(  ��@��*=  ��p��"v����� $ h�  $��}  �   ������j �� ��  n�   ���  �   ~����}������j �� ��  l�   @ l�   ���  �   ~����}������j �� ��  l�     l�   ���  �   ~����}������j �� ��  l�    l�   ���  �   ~����}������j �� ��  l�    l�   ���  �   ~����}�$������j �� ��  l�    l�  ���  �   ~��������  l�   � ��  ���� � &  � �}j ��~��  ��!��  �� #��� ~�(   � �}  ��  ~�� ��	  h�" �  �l�    ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��  ��       �                           0      
       T       ���o    �                                    Linker: AMD LLD 18.0.0 AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.1 24491 1e0fda770a2079fbd71e4b70974d74f62fd3af10)                                T     �6                         �
          �      @       8    
 0G              .note .dynsym .gnu.hash .hash .dynstr .rodata .text .dynamic .relro_padding .bss .comment .symtab .shstrtab .strtab  gemv_nt_bf16xfp8_block_v2 gemv_nt_bf16xfp8_block_v2.kd __hip_cuid_2751b0aab6457f19 _DYNAMIC                                                                                    8      8      D                                          �      �      `                              ���o       �      �      (                                                      (                                         0      0      T                              '             �      �      @               @               /                           �                             5             �6      �      p                            >             07      0      �                             M             0G      0                                    R      0               0      �                             [                      �      x                           c                      P      u                              m                      �      ]                               